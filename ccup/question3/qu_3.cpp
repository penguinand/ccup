#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>

#define INPUT_DIM     10
#define HIDDEN_DIM    32
#define OUTPUT_DIM    1
#define BATCH_SIZE    256
#define EPOCHS        200
#define LR            1e-3      // base learning rate for Adam
#define B1            0.9       // Adam beta1
#define B2            0.999     // Adam beta2
#define EPS           1e-8

#define TILE 16

#define HIP_CHECK(cmd) do {                            \
    hipError_t e = cmd;                                \
    if (e != hipSuccess) {                             \
      std::cerr<<#cmd<<" failed: "<<hipGetErrorString(e)<<std::endl; \
      std::exit(-1);                                   \
    }                                                   \
} while(0)

//------------------------------------------------------------------------------
// GEMM kernel: C[MxN] = A[MxK] * B[KxN]
__global__ void matmul(const double* A, const double* B, double* C,
                      int M, int N, int K) {
    __shared__ double sA[TILE][TILE], sB[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;
    double sum = 0;
    for (int t = 0; t < (K+TILE-1)/TILE; ++t) {
        int aRow = Row, aCol = t*TILE+tx;
        int bRow = t*TILE+ty, bCol = Col;
        sA[ty][tx] = (aRow<M && aCol<K) ? A[aRow*K + aCol] : 0.0;
        sB[ty][tx] = (bRow<K && bCol<N) ? B[bRow*N + bCol] : 0.0;
        __syncthreads();
        for(int i=0;i<TILE;i++) sum += sA[ty][i] * sB[i][tx];
        __syncthreads();
    }
    if (Row<M && Col<N) C[Row*N + Col] = sum;
}

// add bias and ReLU in-place on Z[MxN]
__global__ void add_bias_relu(double* Z, const double* b, int M, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tot = M*N;
    if (idx<tot) {
        int j = idx % N;
        double v = Z[idx] + b[j];
        Z[idx] = (v>0.0? v : 0.0);
    }
}

// add bias only
__global__ void add_bias(double* Z, const double* b, int M, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tot = M*N;
    if (idx<tot) {
        int j = idx%N;
        Z[idx] += b[j];
    }
}

// compute dLoss/dPred = 2*(pred - target)/batch
__global__ void comp_dout(const double* pred, const double* tgt,
                          double* dOut, int batch){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<batch) dOut[i] = 2.0*(pred[i]-tgt[i])/batch;
}

// ReLU backward
__global__ void relu_bwd(double* grad, const double* act, int M, int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx< M*N && act[idx]<=0.0) grad[idx]=0.0;
}

// reduce sum over rows, output length N
__global__ void reduce_col(const double* X, double* out, int M, int N){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j<N){
        double s=0;
        for(int i=0;i<M;i++) s += X[i*N+j];
        out[j] = s;
    }
}

// Adam 更新 kernel
__global__ void adam_update(double* w, const double* g,
                            double* m, double* v,
                            double lr, double b1, double b2,
                            double eps, int t, int sz)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<sz){
        double gi = g[i];
        m[i] = b1*m[i] + (1-b1)*gi;
        v[i] = b2*v[i] + (1-b2)*gi*gi;
        double mhat = m[i] / (1 - pow(b1, t));
        double vhat = v[i] / (1 - pow(b2, t));
        w[i] -= lr * mhat / (sqrt(vhat)+eps);
    }
}

// load JSON array of numbers
std::vector<double> load_json(const std::string& fn){
    std::ifstream in(fn);
    if(!in){ std::cerr<<"Open "<<fn<<" failed\n"; exit(-1); }
    std::string s((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
    for(auto& c:s) if(!(isdigit(c)||c=='.'||c=='-')) c=' ';
    std::stringstream ss(s);
    std::vector<double> a; double v;
    while(ss>>v) a.push_back(v);
    return a;
}

// normalize to [0,1]
void normalize(std::vector<double>& a, double& mn, double& mx){
    mn = *std::min_element(a.begin(), a.end());
    mx = *std::max_element(a.begin(), a.end());
    for(auto& v:a) v=(v-mn)/(mx-mn);
}

// denormalize
void denorm(std::vector<double>& a, double mn, double mx){
    for(auto& v:a) v=v*(mx-mn)+mn;
}

// slide window
void make_dataset(const std::vector<double>& data,
                  std::vector<double>& X, std::vector<double>& Y){
    int S = int(data.size())-INPUT_DIM;
    X.resize(S*INPUT_DIM);
    Y.resize(S);
    for(int i=0;i<S;i++){
        for(int j=0;j<INPUT_DIM;j++)
            X[i*INPUT_DIM+j]=data[i+j];
        Y[i]=data[i+INPUT_DIM];
    }
}

int main(){
    // 1) load & normalize
    auto raw = load_json("starlink_bw.json");
    double mn,mx;
    normalize(raw, mn,mx);
    // 2) make samples
    std::vector<double> X, Y;
    make_dataset(raw, X, Y);
    int S = Y.size();
    // 3) shuffle & split
    std::vector<int> idx(S);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(1234);
    std::shuffle(idx.begin(), idx.end(), rng);
    int ntr = S*8/10, nte=S-ntr;
    std::vector<double> Xtr(ntr*INPUT_DIM), Ytr(ntr),
                        Xte(nte*INPUT_DIM), Yte(nte);
    for(int i=0;i<ntr;i++){
        int k=idx[i];
        std::copy(&X[k*INPUT_DIM], &X[k*INPUT_DIM+INPUT_DIM], &Xtr[i*INPUT_DIM]);
        Ytr[i]=Y[k];
    }
    for(int i=0;i<nte;i++){
        int k=idx[ntr+i];
        std::copy(&X[k*INPUT_DIM], &X[k*INPUT_DIM+INPUT_DIM], &Xte[i*INPUT_DIM]);
        Yte[i]=Y[k];
    }
    std::cout<<"Total="<<S<<" train="<<ntr<<" test="<<nte<<"\n";

    // 4) alloc weights/bias and Adam moments
    int szW1=INPUT_DIM*HIDDEN_DIM, szB1=HIDDEN_DIM;
    int szW2=HIDDEN_DIM*OUTPUT_DIM, szB2=OUTPUT_DIM;
    double *d_W1,*d_b1,*d_W2,*d_b2;
    double *d_mW1,*d_vW1,*d_mb1,*d_vb1;
    double *d_mW2,*d_vW2,*d_mb2,*d_vb2;
    HIP_CHECK( hipMalloc(&d_W1, szW1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_b1, szB1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_W2, szW2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_b2, szB2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_mW1, szW1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_vW1, szW1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_mb1, szB1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_vb1, szB1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_mW2, szW2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_vW2, szW2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_mb2, szB2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_vb2, szB2*sizeof(double)) );
    // init host parameters
    std::vector<double> hW1(szW1), hb1(szB1,0.0),
                        hW2(szW2), hb2(szB2,0.0);
    std::normal_distribution<double> norm(0,0.1);
    for(auto&v:hW1) v = norm(rng);
    for(auto&v:hW2) v = norm(rng);
    HIP_CHECK( hipMemcpy(d_W1, hW1.data(), szW1*sizeof(double), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_b1, hb1.data(), szB1*sizeof(double), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_W2, hW2.data(), szW2*sizeof(double), hipMemcpyHostToDevice) );
    HIP_CHECK( hipMemcpy(d_b2, hb2.data(), szB2*sizeof(double), hipMemcpyHostToDevice) );
    // zero Adam moments
    HIP_CHECK( hipMemset(d_mW1,0,szW1*sizeof(double)) );
    HIP_CHECK( hipMemset(d_vW1,0,szW1*sizeof(double)) );
    HIP_CHECK( hipMemset(d_mb1,0,szB1*sizeof(double)) );
    HIP_CHECK( hipMemset(d_vb1,0,szB1*sizeof(double)) );
    HIP_CHECK( hipMemset(d_mW2,0,szW2*sizeof(double)) );
    HIP_CHECK( hipMemset(d_vW2,0,szW2*sizeof(double)) );
    HIP_CHECK( hipMemset(d_mb2,0,szB2*sizeof(double)) );
    HIP_CHECK( hipMemset(d_vb2,0,szB2*sizeof(double)) );

    // 5) batch buffers + grads
    double *d_Xb,*d_Z1,*d_A1,*d_pred,*d_yb;
    double *d_dOut,*d_dW2,*d_db2,*d_dA1,*d_dZ1,*d_dW1,*d_db1;
    HIP_CHECK( hipMalloc(&d_Xb,  BATCH_SIZE*INPUT_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_Z1,  BATCH_SIZE*HIDDEN_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_A1,  BATCH_SIZE*HIDDEN_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_pred,BATCH_SIZE*OUTPUT_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_yb,  BATCH_SIZE*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_dOut,BATCH_SIZE*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_dW2, szW2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_db2, szB2*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_dA1, BATCH_SIZE*HIDDEN_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_dZ1, BATCH_SIZE*HIDDEN_DIM*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_dW1, szW1*sizeof(double)) );
    HIP_CHECK( hipMalloc(&d_db1, szB1*sizeof(double)) );

    int nBatch = (ntr+BATCH_SIZE-1)/BATCH_SIZE;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 6) training loop
    for(int ep=1; ep<=EPOCHS; ep++){
        double lossE=0;
        for(int b=0;b<nBatch;b++){
            int off=b*BATCH_SIZE;
            int sz = std::min(BATCH_SIZE, ntr-off);
            // copy batch
            HIP_CHECK( hipMemcpy(d_Xb, &Xtr[off*INPUT_DIM],
                        sz*INPUT_DIM*sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK( hipMemcpy(d_yb, &Ytr[off],
                        sz*sizeof(double), hipMemcpyHostToDevice));

            // forward 1: Z1 = Xb*W1
            dim3 g1((HIDDEN_DIM+TILE-1)/TILE,(sz+TILE-1)/TILE), b1(TILE,TILE);
            hipLaunchKernelGGL(matmul, g1,b1,0,0,
                               d_Xb,d_W1,d_Z1, sz,HIDDEN_DIM,INPUT_DIM);
            // +b1+ReLU
            {
                int T=sz*HIDDEN_DIM;
                int blk=(T+255)/256;
                add_bias_relu<<<blk,256>>>(d_Z1,d_b1,sz,HIDDEN_DIM);
                HIP_CHECK( hipMemcpy(d_A1,d_Z1,T*sizeof(double),hipMemcpyDeviceToDevice) );
            }
            // forward 2: pred = A1*W2 + b2
            dim3 g2((OUTPUT_DIM+TILE-1)/TILE,(sz+TILE-1)/TILE);
            hipLaunchKernelGGL(matmul, g2,b1,0,0,
                               d_A1,d_W2,d_pred, sz,OUTPUT_DIM,HIDDEN_DIM);
            {
                int T=sz*OUTPUT_DIM;
                int blk=(T+255)/256;
                add_bias<<<blk,256>>>(d_pred,d_b2,sz,OUTPUT_DIM);
            }
            // loss grad
            {
                int blk=(sz+255)/256;
                comp_dout<<<blk,256>>>(d_pred,d_yb,d_dOut,sz);
            }
            // dW2 = A1^T * dOut
            dim3 g3((OUTPUT_DIM+TILE-1)/TILE,(HIDDEN_DIM+TILE-1)/TILE);
            hipLaunchKernelGGL(matmul, g3,b1,0,0,
                               d_A1,d_dOut,d_dW2, HIDDEN_DIM,OUTPUT_DIM,sz);
            // db2
            reduce_col<<<(OUTPUT_DIM+255)/256,256>>>(d_dOut,d_db2,sz,OUTPUT_DIM);
            // dA1 = dOut * W2^T
            dim3 g4((HIDDEN_DIM+TILE-1)/TILE,(sz+TILE-1)/TILE);
            hipLaunchKernelGGL(matmul, g4,b1,0,0,
                               d_dOut,d_W2,d_dA1, sz,HIDDEN_DIM,OUTPUT_DIM);
            // relu backward
            {
                int T=sz*HIDDEN_DIM, blk=(T+255)/256;
                relu_bwd<<<blk,256>>>(d_dA1,d_Z1,sz,HIDDEN_DIM);
                HIP_CHECK( hipMemcpy(d_dZ1,d_dA1,T*sizeof(double),hipMemcpyDeviceToDevice) );
            }
            // dW1 = Xb^T * dZ1
            dim3 g5((HIDDEN_DIM+TILE-1)/TILE,(INPUT_DIM+TILE-1)/TILE);
            hipLaunchKernelGGL(matmul, g5,b1,0,0,
                               d_Xb,d_dZ1,d_dW1, INPUT_DIM,HIDDEN_DIM,sz);
            // db1
            reduce_col<<<(HIDDEN_DIM+255)/256,256>>>(d_dZ1,d_db1,sz,HIDDEN_DIM);

            // Adam update all params
            int t = ep;
            // W1,b1
            {
                int cw=(szW1+255)/256, cb=(szB1+255)/256;
                adam_update<<<cw,256>>>(d_W1,d_dW1,d_mW1,d_vW1,
                                        LR,B1,B2,EPS,t,szW1);
                adam_update<<<cb,256>>>(d_b1,d_db1,d_mb1,d_vb1,
                                        LR,B1,B2,EPS,t,szB1);
            }
            // W2,b2
            {
                int cw=(szW2+255)/256, cb=(szB2+255)/256;
                adam_update<<<cw,256>>>(d_W2,d_dW2,d_mW2,d_vW2,
                                        LR,B1,B2,EPS,t,szW2);
                adam_update<<<cb,256>>>(d_b2,d_db2,d_mb2,d_vb2,
                                        LR,B1,B2,EPS,t,szB2);
            }
            HIP_CHECK( hipDeviceSynchronize() );

            // accumulate loss on host
            {
                std::vector<double> hp(sz), ht(sz);
                HIP_CHECK( hipMemcpy(hp.data(), d_pred, sz*sizeof(double),
                                     hipMemcpyDeviceToHost) );
                HIP_CHECK( hipMemcpy(ht.data(), d_yb, sz*sizeof(double),
                                     hipMemcpyDeviceToHost) );
                double L=0;
                for(int i=0;i<sz;i++){ double e=hp[i]-ht[i]; L+=e*e; }
                lossE += L/sz;
            }
        }
        if(ep%20==0)
            std::cout<<"Epoch "<<ep<<"/"<<EPOCHS
                     <<" avg MSE="<<lossE/nBatch<<"\n";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_train = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout<<"[TRAIN] "<<t_train<<" ms, throughput "
             <<ntr*EPOCHS/(t_train/1000)<<" samp/s\n";

    // 7) inference
    double t_inf=0;
    std::vector<double> Yp(nte);
    int nb2 = (nte+BATCH_SIZE-1)/BATCH_SIZE;
    for(int b=0;b<nb2;b++){
        int off=b*BATCH_SIZE, sz = std::min(BATCH_SIZE, nte-off);
        HIP_CHECK( hipMemcpy(d_Xb, &Xte[off*INPUT_DIM],
                    sz*INPUT_DIM*sizeof(double), hipMemcpyHostToDevice) );
        auto ti0 = std::chrono::high_resolution_clock::now();
        // forward exactly same as above
        dim3 g1((HIDDEN_DIM+TILE-1)/TILE,(sz+TILE-1)/TILE), b1(TILE,TILE);
        hipLaunchKernelGGL(matmul, g1,b1,0,0,
                           d_Xb,d_W1,d_Z1, sz,HIDDEN_DIM,INPUT_DIM);
        {
            int T=sz*HIDDEN_DIM;
            int blk=(T+255)/256;
            add_bias_relu<<<blk,256>>>(d_Z1,d_b1,sz,HIDDEN_DIM);
            HIP_CHECK( hipMemcpy(d_A1,d_Z1,T*sizeof(double),hipMemcpyDeviceToDevice) );
        }
        dim3 g2((OUTPUT_DIM+TILE-1)/TILE,(sz+TILE-1)/TILE);
        hipLaunchKernelGGL(matmul, g2,b1,0,0,
                           d_A1,d_W2,d_pred, sz,OUTPUT_DIM,HIDDEN_DIM);
        {
            int T=sz*OUTPUT_DIM;
            int blk=(T+255)/256;
            add_bias<<<blk,256>>>(d_pred,d_b2,sz,OUTPUT_DIM);
        }
        HIP_CHECK( hipDeviceSynchronize() );
        auto ti1 = std::chrono::high_resolution_clock::now();
        t_inf += std::chrono::duration<double,std::milli>(ti1-ti0).count();
        HIP_CHECK( hipMemcpy(&Yp[off], d_pred, sz*sizeof(double),
                             hipMemcpyDeviceToHost) );
    }
    std::cout<<"[INFER] "<<t_inf<<" ms, throughput "
             <<nte/(t_inf/1000)<<" samp/s\n";

    // 8) denorm & eval
    denorm(Yp,mn,mx); denorm(Yte,mn,mx);
    double mse=0,mae=0;
    for(int i=0;i<nte;i++){
        double e=Yp[i]-Yte[i];
        mse+=e*e; mae+=fabs(e);
    }
    mse/=nte; mae/=nte;
    std::cout<<"[RESULT] MSE="<<mse<<" MAE="<<mae<<"\n";

    return 0;
}