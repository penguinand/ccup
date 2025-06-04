#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include "FusionKernel.hh"

int main(){
  // 1. 参数
  const int B = 1024, I = 10, H = 20, O = 5;
  // host 数据
  std::vector<float>   h_X (B*I),   h_W1 (I*H),  h_b1 (H),
                       h_W2 (H*O),   h_b2 (O);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &x: h_X ) x = dist(rng);
  for(auto &x: h_W1) x = dist(rng);
  for(auto &x: h_b1) x = dist(rng);
  for(auto &x: h_W2) x = dist(rng);
  for(auto &x: h_b2) x = dist(rng);

  // 转 half
  std::vector<__half> h_Xh(B*I), h_W1h(I*H), h_W2h(H*O);
  for(int i=0;i<B*I;  i++) h_Xh[i]  = __float2half(h_X[i]);
  for(int i=0;i<I*H;  i++) h_W1h[i] = __float2half(h_W1[i]);
  for(int i=0;i<H*O;  i++) h_W2h[i] = __float2half(h_W2[i]);

  // 输出 buffer (page-locked)
  float *h_Y;
  hipHostMalloc(&h_Y, sizeof(float)*B*O, hipHostMallocDefault);

  // 2. Device malloc + copy
  __half *d_Xh, *d_W1, *d_W2;
  float   *d_Y;
  hipMalloc(&d_Xh, sizeof(__half)*B*I);
  hipMalloc(&d_W1,  sizeof(__half)*I*H);
  hipMalloc(&d_W2,  sizeof(__half)*H*O);
  hipMalloc(&d_Y,   sizeof(float)*B*O);
  hipMemcpy(d_Xh, h_Xh.data(), sizeof(__half)*B*I, hipMemcpyHostToDevice);
  hipMemcpy(d_W1,  h_W1h.data(),sizeof(__half)*I*H, hipMemcpyHostToDevice);
  hipMemcpy(d_W2,  h_W2h.data(),sizeof(__half)*H*O, hipMemcpyHostToDevice);
  // 拷贝常量 Bias
  hipMemcpyToSymbol(HIP_SYMBOL(d_b1), h_b1.data(), sizeof(float)*H);
  hipMemcpyToSymbol(HIP_SYMBOL(d_b2), h_b2.data(), sizeof(float)*O);

  // 3. 构建 GraphCapture
  FusionParams fp{d_Xh,d_W1,d_W2,d_Y,B,I,H,O};
  hipStream_t  s;
  hipStreamCreate(&s);
  hipGraph_t   graph;
  hipGraphCreate(&graph, 0);
  hipGraphNode_t node;
  hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);

    dim3 grid( (O + TB_N - 1)/TB_N,
               (B + TB_M - 1)/TB_M );
    dim3 block( (TB_M/TILE_M)*(TB_N/TILE_N) );
    hipLaunchKernelGGL(FusionKernel,
                       grid, block, 0, s,
                       fp);

  hipStreamEndCapture(s, &graph);
  hipGraphExec_t exec;
  hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

  // 4. Warmup + 性能测试
  hipGraphLaunch(exec, s);
  hipStreamSynchronize(s);

  hipEvent_t e1,e2;
  hipEventCreate(&e1);
  hipEventCreate(&e2);
  hipEventRecord(e1, s);
    const int NITER = 1000;
    for(int i=0;i<NITER;i++){
      hipGraphLaunch(exec, s);
    }
  hipEventRecord(e2, s);
  hipEventSynchronize(e2);
  float ms=0.f;
  hipEventElapsedTime(&ms, e1, e2);
  double ops = double(B) * (double(I)*H + double(H)*O) * 2.0;
  std::cout<<"Avg latency: "<<ms/NITER<<" ms, "
           <<(ops*1e-9)/(ms/NITER*1e-3)<<" TFLOPS\n";

  // 5. 拷回结果并简单验证
  hipMemcpy(h_Y, d_Y, sizeof(float)*B*O, hipMemcpyDeviceToHost);
  std::cout<<"Y[0..4] = ";
  for(int i=0;i<5;i++) std::cout<<h_Y[i*O]<<" ";
  std::cout<<"\n";

  return 0;
}