#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define N 1024
#define M 2048
#define P 512
#define TILE 16

__global__ void matmul_kernel(const double * __restrict__ A,
                              const double * __restrict__ B,
                              double * __restrict__ C,
                              int n,int m,int p) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    double sum = 0;
    for (int t = 0; t < (m+TILE-1)/TILE; ++t) {
        int a_col = t*TILE + threadIdx.x;
        int b_row = t*TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = 
            (row<n && a_col<m ? A[row*m + a_col] : 0.0);
        Bs[threadIdx.y][threadIdx.x] = 
            (b_row<m && col<p ? B[b_row*p + col] : 0.0);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row<n && col<p) C[row*p + col] = sum;
}

void init_matrix(std::vector<double>& X, int n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> d(-1.0,1.0);
    for (int i = 0; i < n; ++i) X[i] = d(gen);
}

bool validate(const std::vector<double>& A,
              const std::vector<double>& B) {
    for (int i = 0, e = N*P; i < e; ++i)
        if (fabs(A[i]-B[i]) > 1e-6) return false;
    return true;
}

void matmul_cpu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C) {
    for (int i=0;i<N;i++)
      for(int j=0;j<P;j++){
        double s=0;
        for(int k=0;k<M;k++)
          s+=A[i*M+k]*B[k*P+j];
        C[i*P+j]=s;
      }
}

int main(){
    std::vector<double> A(N*M), B(M*P), C(N*P), C_ref(N*P);
    init_matrix(A,N*M);
    init_matrix(B,M*P);
    matmul_cpu(A,B,C_ref);

    double *dA,*dB,*dC;
    hipMalloc(&dA,sizeof(double)*N*M);
    hipMalloc(&dB,sizeof(double)*M*P);
    hipMalloc(&dC,sizeof(double)*N*P);
    hipMemcpy(dA,A.data(),sizeof(double)*N*M,hipMemcpyHostToDevice);
    hipMemcpy(dB,B.data(),sizeof(double)*M*P,hipMemcpyHostToDevice);

    dim3 block(TILE,TILE), grid((P+TILE-1)/TILE,(N+TILE-1)/TILE);
    hipEvent_t st, ed;
    hipEventCreate(&st); hipEventCreate(&ed);

    hipEventRecord(st);
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0,
                       dA,dB,dC,N,M,P);
    hipEventRecord(ed);
    hipEventSynchronize(ed);

    float ms;
    hipEventElapsedTime(&ms, st, ed);
    std::cout << "[HIP] Elapsed " << ms << " ms\n";

    hipMemcpy(C.data(), dC, sizeof(double)*N*P, hipMemcpyDeviceToHost);
    std::cout << "[HIP] Valid=" << validate(C_ref,C) << "\n";

    hipFree(dA); hipFree(dB); hipFree(dC);
    hipEventDestroy(st); hipEventDestroy(ed);
    return 0;
}
