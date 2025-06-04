#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// -----------------------------------------------------------------------------
// Tuneable Tile 参数
static constexpr int TB_M   = 32;   // block 输出行数 (batch 维)
static constexpr int TB_N   = 32;   // block 输出列数 (out dim)
static constexpr int TB_K   = 8;    // block K 方向分块
static constexpr int TM     = 2;     // 每线程负责行数
static constexpr int TN     = 2;     // 每线程负责列数
static constexpr int TILE_M = TM;
static constexpr int TILE_N = TN;
// 确保 (TB_M/TM)*(TB_N/TN) == blockDim.x

// 常量 Bias
__constant__ float d_b1[1024];  // H_DIM 上限示例 1024
__constant__ float d_b2[1024];  // O_DIM 上限示例 1024

struct FusionParams {
    const __half * __restrict__ d_Xh;  // [I_DIM × BATCH]^T
    const __half * __restrict__ d_W1;  // [I_DIM × H_DIM]
    const __half * __restrict__ d_W2;  // [H_DIM × O_DIM]
    float *        __restrict__ d_Y;   // [BATCH × O_DIM]
    int batch, idim, hdim, odim;
};

// Shared Memory Buffers
// 载入一个 (TB_K × TB_M) 和 (TB_K × TB_N) 的 tile
extern __shared__ __half smem[];

// smem 布局：
//  [0 .. TB_K*TB_M)       存 A_tile = X^T[k0..k0+TB_K, bm..bm+TB_M]
//  [TB_K*TB_M .. end)     存 B_tile = W2[k0..k0+TB_K, bn..bn+TB_N]
//
// 大小 = TB_K*TB_M + TB_K*TB_N
// launch 时指定 sharemem = sizeof(__half)*(TB_K*TB_M + TB_K*TB_N)

__global__ void FusionKernel(const FusionParams p) {
    // Block tile 基址
    int bm = blockIdx.y * TB_M;
    int bn = blockIdx.x * TB_N;

    // 线程在 tile 内的索引 (ty, tx)
    int ntx = TB_N / TN;
    int ty  = threadIdx.x / ntx;        // 0..TB_M/TM
    int tx  = threadIdx.x % ntx;        // 0..TB_N/TN

    // 寄存器 tile
    float rY[TM][TN];
    // 初始化 0
    #pragma unroll
    for(int i=0;i<TM;i++)
        for(int j=0;j<TN;j++)
            rY[i][j] = 0.f;

    // 指向 Shared Memory 区
    __half *A_s = smem;
    __half *B_s = smem + TB_K*TB_M;

    // K 方向分块
    for(int k0 = 0; k0 < p.hdim; k0 += TB_K) {
        int Kblk = min(TB_K, p.hdim - k0);

        // 1) 全 Block 合作把 A_tile 和 B_tile load 到 smem
        //    A_tile: shape (Kblk x TB_M) from Xh^T
        //    B_tile: shape (Kblk x TB_N) from W2
        int nThreads = blockDim.x;
        int elemsA = Kblk * TB_M;
        int elemsB = Kblk * TB_N;
        int total  = elemsA + elemsB;
        int perTh  = (total + nThreads - 1)/nThreads;

        int base = threadIdx.x * perTh;
        int lim  = min(perTh, total-base);
        for(int i=0;i<lim;i++){
            int idx = base + i;
            if(idx < elemsA){
                // load A
                int kk = idx / TB_M;
                int mm = idx % TB_M;
                int gi = bm + mm;
                int hi = k0 + kk;
                A_s[idx] = (gi<p.batch && hi<p.hdim)
                         ? p.d_Xh[hi * p.batch + gi]
                         : __float2half(0.f);
            } else {
                // load B
                int idxB = idx - elemsA;
                int kk = idxB / TB_N;
                int nn = idxB % TB_N;
                int hi = k0 + kk;
                int go = bn + nn;
                B_s[idxB] = (hi<p.hdim && go<p.odim)
                          ? p.d_W2[hi * p.odim + go]
                          : __float2half(0.f);
            }
        }
        __syncthreads();

        // 2) 对 Kblk 方向做累加
        //    并在累加时 inline bias1 + ReLU
        #pragma unroll
        for(int kk=0; kk<Kblk; kk++){
            // 从 Shared Memory 读 A_rb 对应 TM 个半精度
            // 和 B_rb 对应 TN 个半精度
            // 并转换为 f32
            float Areg[TM], Breg[TN];
            #pragma unroll
            for(int i=0;i<TM;i++){
                int mm = ty*TM + i;
                Areg[i] = __half2float( A_s[ kk*TB_M + mm ] );
            }
            #pragma unroll
            for(int j=0;j<TN;j++){
                int nn = tx*TN + j;
                Breg[j] = __half2float( B_s[ kk*TB_N + nn ] );
            }
            // bias1 & ReLU 只做一次
            float bias1 = d_b1[k0+kk];
            #pragma unroll
            for(int i=0;i<TM;i++){
                float a = Areg[i]*1.f + bias1;
                // ReLU
                Areg[i] = a>0.f ? a : 0.f;
            }
            // 矩阵乘累加
            #pragma unroll
            for(int i=0;i<TM;i++){
                #pragma unroll
                for(int j=0;j<TN;j++){
                    // TODO: 在此处替换为 xDLOPS / WMMA intrinsic
                    rY[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        __syncthreads();
    }

    // 3) Epilogue：写回
    #pragma unroll
    for(int i=0;i<TM;i++){
        int gi = bm + ty*TM + i;
        if(gi >= p.batch) continue;
        #pragma unroll
        for(int j=0;j<TN;j++){
            int go = bn + tx*TN + j;
            if(go >= p.odim) continue;
            float v = rY[i][j] + d_b2[go];
            p.d_Y[gi * p.odim + go] = v;
        }
    }

    return;
}
