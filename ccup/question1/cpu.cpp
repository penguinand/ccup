#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Problem size
static const int N = 1024, M = 2048, P = 512;
// Block sizes
static const int B2 = 256;  // L2 度量块
static const int B1 = 64;   // L1 度量块

void init_matrix(std::vector<double>& A, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0, e = rows*cols; i < e; ++i) A[i] = dist(gen);
}

// 校验
bool validate(const std::vector<double>& A,
              const std::vector<double>& B) {
    for (int i = 0, e = N*P; i < e; ++i)
        if (std::abs(A[i] - B[i]) > 1e-6) return false;
    return true;
}

// Baseline 单线程
void matmul_baseline(const double *A, const double *B, double *C) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < P; ++j) {
        double s = 0;
        for (int k = 0; k < M; ++k)
            s += A[i*M + k] * B[k*P + j];
        C[i*P + j] = s;
      }
}

// Opt：2 级 Block + OpenMP + SIMD + Prefetch
void matmul_opt(const double *A, const double *B, double *C) {
    // 初始化 C
    #pragma omp parallel for schedule(static)
    for (int i = 0, e = N*P; i < e; ++i) C[i] = 0.0;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i2 = 0; i2 < N; i2 += B2)
    for (int j2 = 0; j2 < P; j2 += B2) {
        for (int k2 = 0; k2 < M; k2 += B2) {
            int i2_end = std::min(i2 + B2, N);
            int j2_end = std::min(j2 + B2, P);
            int k2_end = std::min(k2 + B2, M);

            // L1Blocking
            for (int i1 = i2; i1 < i2_end; i1 += B1)
            for (int j1 = j2; j1 < j2_end; j1 += B1) {
                int i1_end = std::min(i1 + B1, i2_end);
                int j1_end = std::min(j1 + B1, j2_end);

                for (int k1 = k2; k1 < k2_end; k1 += B1) {
                    int k1_end = std::min(k1 + B1, k2_end);

                    for (int i = i1; i < i1_end; ++i) {
                        const double *Ai = A + i*M;
                        double *Ci = C + i*P;
                        for (int k = k1; k < k1_end; ++k) {
                            // 软件预取下一行数据
                            _mm_prefetch((const char*)(Ai + k + 16), _MM_HINT_T0);
                            double a = Ai[k];
                            const double *Bk = B + k*P;
                            // SIMD 化内层 j 循环
                            #pragma omp simd aligned(Ci,Bk:64)
                            for (int j = j1; j < j1_end; ++j) {
                                Ci[j] += a * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    std::vector<double> A(N*M), B(M*P), C1(N*P), C2(N*P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);

    double t0 = omp_get_wtime();
    matmul_baseline(A.data(), B.data(), C1.data());
    double t_base = omp_get_wtime() - t0;
    std::cout << "[Baseline] " << t_base << " s\n";

    t0 = omp_get_wtime();
    matmul_opt(A.data(), B.data(), C2.data());
    double t_opt = omp_get_wtime() - t0;
    std::cout << "[Optimized] " << t_opt
              << " s, speedup=" << t_base/t_opt
              << ", valid=" << validate(C1,C2) << "\n";
    return 0;
}
