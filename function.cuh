#include <iostream>
#include <math.h>
#include <cuComplex.h>
#include <complex>
#include <string>
#include <random>
#include <ctime>
#include "error.cuh"
#include <chrono>
#include "cufft.h"
#include <cufftXt.h>
#include <iomanip>
#include "cudaArray_4dim.cuh"
#include <fstream>
// #include <gmp.h>
using namespace std;

typedef std::complex<double> Complex;
#define PI acos(-1)
static const int64_t _two31 = INT64_C(1) << 31; // 2^31
static const int64_t _two32 = INT64_C(1) << 32; // 2^32
static const double _two63 = pow(2, 63); // 2^63
static const double _two64 = pow(2, 64); // 2^64
typedef uint32_t Torus32;
#define EPSILON 1e-15
#define DOUBLE_TO_UINT64(d) ((*(uint64_t *)(&(d))) ^ (-( (*(uint64_t *)(&(d))) >> 63) | 0x8000000000000000ULL))




__host__ cuDoubleComplex cexp(const cuDoubleComplex &z)
{
    Complex stl_complex(cuCreal(z), cuCimag(z));
    stl_complex = exp(stl_complex);
    return make_cuDoubleComplex(real(stl_complex), imag(stl_complex));
}



class Params128
{
public:
    int N;
    int n;
    int nbar;  // 用于tlwe2tgsw转换
    int bk_l;
    int bk_Bgbit;
    int bk_lbar;    //
    int bk_Bgbitbar;    //
    int ks_t;  
    int ks_basebit; 
    int ks_tbar;
    int ks_basebitbar;
    double ks_stdev;
    double bk_stdev;
    double ks_stdevbar;
    double bk_stdevbar;
    int Bg;
    int Bgbar;    
    double *H;
    double *Hbar;   //
    uint32_t offset;
    uint64_t offsetbar; //
    int *decbit;
    int *decbitbar;     //
    cuDoubleComplex *twist;
    cuDoubleComplex *twistlong;     //
    int Msize;
    uint32_t inter;
    uint64_t interlong;


public:
    Params128(int N, int n, int nbar, int bk_l, int bk_Bgbit, int bk_lbar, int bk_Bgbitbar, int ks_t, int ks_basebit, int ks_tbar, int ks_basebitbar, double ks_stdev, double bk_stdev, double ks_stdevbar, double bk_stdevbar, int Msize) : N(N),
                                                                                                 n(n),
                                                                                                 nbar(nbar),
                                                                                                 bk_l(bk_l),
                                                                                                 bk_Bgbit(bk_Bgbit),                                                                                                 
                                                                                                 bk_lbar(bk_lbar),
                                                                                                 bk_Bgbitbar(bk_Bgbitbar),
                                                                                                 ks_t(ks_t),
                                                                                                 ks_basebit(ks_basebit),                                                                                               
                                                                                                 ks_tbar(ks_tbar),
                                                                                                 ks_basebitbar(ks_basebitbar),
                                                                                                 ks_stdev(ks_stdev),
                                                                                                 bk_stdev(bk_stdev),
                                                                                                 ks_stdevbar(ks_stdevbar),
                                                                                                 bk_stdevbar(bk_stdevbar),
                                                                                                 Bg(1 << bk_Bgbit),
                                                                                                 Bgbar(1 << bk_Bgbitbar),
                                                                                                 Msize(Msize),
                                                                                                 inter(_two31 / Msize * 2),
                                                                                                 interlong(_two63 / Msize * 2)
    {
        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        this->Hbar = (double *)malloc(sizeof(double) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->Hbar[i] = pow(this->Bgbar, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        this->offsetbar = 0;
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->offsetbar += pow(2, 64) * this->Hbar[i];
        }
        this->offsetbar = this->Bgbar / 2 * this->offsetbar;

        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }

        this->decbitbar = (int *)malloc(sizeof(int) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->decbitbar[i] = 64 - (i + 1) * this->bk_Bgbitbar;
        }
        
        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            // attention!
            // twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
            twist[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->N));
        }

        this->twistlong = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->nbar / 2);
        for (int k = 0; k < this->nbar / 2; k++)
        {
            twistlong[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->nbar));
        }
    }
    Params128(const Params128 &p128)
    {
        this->N = p128.N;
        this->n = p128.n;
        this->nbar = p128.nbar;
        this->bk_l = p128.bk_l;
        this->bk_Bgbit = p128.bk_Bgbit;
        this->bk_lbar = p128.bk_lbar;
        this->bk_Bgbitbar = p128.bk_Bgbitbar;
        this->ks_t = p128.ks_t;
        this->ks_basebit = p128.ks_basebit;
        this->ks_tbar = p128.ks_tbar;
        this->ks_basebitbar = p128.ks_basebitbar;
        this->ks_stdev = p128.ks_stdev;
        this->bk_stdev = p128.bk_stdev;
        this->ks_stdevbar = p128.ks_stdevbar;
        this->bk_stdevbar = p128.bk_stdevbar;
        this->Bg = p128.Bg;
        this->Bgbar = p128.Bgbar;
        this->Msize = p128.Msize;
        this->inter = p128.inter;
        this->interlong = p128.interlong;

        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        this->Hbar = (double *)malloc(sizeof(double) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->Hbar[i] = pow(this->Bgbar, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        this->offsetbar = 0;
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->offsetbar += pow(2, 64) * this->Hbar[i];
        }
        this->offsetbar = this->Bgbar / 2 * this->offsetbar;

        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }

        this->decbitbar = (int *)malloc(sizeof(int) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->decbitbar[i] = 64 - (i + 1) * this->bk_Bgbitbar;
        }

        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            // attention!
            // twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
            twist[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->N));
        }

        this->twistlong = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->nbar / 2);
        for (int k = 0; k < this->nbar / 2; k++)
        {
            twistlong[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->nbar));
        }
    }
    Params128 &operator=(const Params128 &p128)
    {
        if (this == &p128)
        {
            return *this;
        }
        this->N = p128.N;
        this->n = p128.n;
        this->nbar = p128.nbar;
        this->bk_l = p128.bk_l;
        this->bk_Bgbit = p128.bk_Bgbit;
        this->bk_lbar = p128.bk_lbar;
        this->bk_Bgbitbar = p128.bk_Bgbitbar;
        this->ks_t = p128.ks_t;
        this->ks_basebit = p128.ks_basebit;
        this->ks_tbar = p128.ks_tbar;
        this->ks_basebitbar = p128.ks_basebitbar;
        this->ks_stdev = p128.ks_stdev;
        this->bk_stdev = p128.bk_stdev;
        this->ks_stdevbar = p128.ks_stdevbar;
        this->bk_stdevbar = p128.bk_stdevbar;
        this->Bg = p128.Bg;
        this->Bgbar = p128.Bgbar;
        this->Msize = p128.Msize;
        this->inter = p128.inter;
        this->interlong = p128.interlong;

        free(this->H);
        this->H = (double *)malloc(sizeof(double) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->H[i] = pow(this->Bg, (-(i + 1)));
        }

        free(this->Hbar);
        this->Hbar = (double *)malloc(sizeof(double) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->Hbar[i] = pow(this->Bgbar, (-(i + 1)));
        }

        this->offset = 0;
        for (int i = 0; i < this->bk_l; i++)
        {
            this->offset += _two32 * this->H[i];
        }
        this->offset = this->Bg / 2 * this->offset;

        this->offsetbar = 0;
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->offsetbar += pow(2, 64) * this->Hbar[i];
        }
        this->offsetbar = this->Bgbar / 2 * this->offsetbar;

        free(this->decbit);
        this->decbit = (int *)malloc(sizeof(int) * this->bk_l);
        for (int i = 0; i < this->bk_l; i++)
        {
            this->decbit[i] = 32 - (i + 1) * this->bk_Bgbit;
        }

        free(this->decbitbar);
        this->decbitbar = (int *)malloc(sizeof(int) * this->bk_lbar);
        for (int i = 0; i < this->bk_lbar; i++)
        {
            this->decbitbar[i] = 64 - (i + 1) * this->bk_Bgbitbar;
        }

        free(this->twist);
        this->twist = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->N / 2);
        for (int k = 0; k < this->N / 2; k++)
        {
            // attention!
            // twist[k] = cexp(make_cuDoubleComplex(0, 2 * double(k) * PI / this->N));
            twist[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->N));
        }

        free(this->twistlong);
        this->twistlong = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * this->nbar / 2);
        for (int k = 0; k < this->nbar / 2; k++)
        {
            twistlong[k] = cexp(make_cuDoubleComplex(0, double(k) * PI / this->nbar));
        }

        return *this;
    }
    ~Params128()
    {
        if (this->H)
        {
            free(this->H);
            this->H = nullptr;
        }
        if (this->Hbar)
        {
            free(this->Hbar);
            this->Hbar = nullptr;
        }
        if (this->decbit)
        {
            free(this->decbit);
            this->decbit = nullptr;
        }
        if (this->decbitbar)
        {
            free(this->decbitbar);
            this->decbitbar = nullptr;
        }
        if (this->twist)
        {
            free(this->twist);
            this->twist = nullptr;
        }
        if (this->twistlong)
        {
            free(this->twistlong);
            this->twistlong = nullptr;
        }
    }
};


double sign(double d)
{
    if (d > EPSILON)
    {
        return 1;
    }
    else if (d < -EPSILON)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}


uint64_t DoubleToUint64(double d)
{
    double a = fmod(d, _two64);
    if(sign(a) == -1)
    {
        return uint64_t(a + _two64);
    }
    else{
        return uint64_t(a);
    }   
}

Torus32 dtot32(double d)
{
    double dsign = sign(d);
    return Torus32(round(fmod(d * dsign, 1) * _two32) * dsign);
}


uint64_t dtot64(double d)
{
    double dsign = sign(d);
    return uint64_t(round(fmod(d * dsign, 1) * _two64) * dsign);
}

uint32_t gaussian32(uint32_t mu, double alpha)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    uint32_t ga = dtot32(dis(gen)) + mu;
    return ga;
}

uint64_t gaussian64(uint64_t mu, double alpha)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    uint64_t ga = dtot64(dis(gen)) + mu;
    return ga;
}


uint32_t mutoT(int mu, int Msize)
{
    return uint32_t(_two31 / Msize * 2 * mu);
}

uint64_t mutoT64(int mu, int Msize)
{
    return uint64_t(_two63 / Msize * 2 * mu);
}

uint32_t Ttomu(uint32_t phase, uint32_t inter)
{
    uint32_t half = uint32_t(inter / 2);
    return uint32_t(uint32_t(phase + half) / inter);
}

uint64_t Ttomu64(uint64_t phase, uint64_t inter)
{
    uint64_t half = uint64_t(inter / 2);
    return uint64_t(uint64_t(phase + half) / inter);
}


void tlweKeyGen(int32_t *tlwekey, int keylen)
{
    srand((int)time(NULL));
    for (int i = 0; i < keylen; i++)
    {
        tlwekey[i] = rand() % 2;
    }
}

void tlweSymEnc(uint32_t mu, int32_t *tlwekey, int keylen, Params128 &p128, uint32_t *c)
{
    uint32_t *a = (uint32_t *)malloc(keylen * sizeof(uint32_t));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    uint32_t product = 0;
    for (int i = 0; i < keylen; i++)
    {
        a[i] = g();
        c[i + 1] = a[i];
        product += a[i] * tlwekey[i];
    }

    uint32_t ga;
    ga = gaussian32(mu, p128.ks_stdev);
    c[0] = (uint32_t)(ga - product);

    free(a);
}

uint32_t tlweSymDec(uint32_t *c, int32_t *tlwekey, int keylen, Params128 &p128)
{
    uint32_t product = 0;
    uint32_t phase = 0;
    for (int i = 0; i < keylen; i++)
    {
        product += c[i + 1] * tlwekey[i];
    }
    phase = c[0] + product;
    uint32_t mu = Ttomu(phase, p128.inter);
    return mu;
}




void gaussian32(uint32_t *vecmu, uint32_t *ga, double alpha, int size = 1)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    for (size_t i = 0; i < size; i++)
    {
        ga[i] = dtot32(dis(gen)) + vecmu[i];
    }
}

void gaussian64(uint64_t *vecmu, uint64_t *ga, double alpha, int size = 1)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, alpha);

    for (size_t i = 0; i < size; i++)
    {
        // ga[i] = dtot64(dis(gen)) + vecmu[i];
        ga[i] = vecmu[i];
    }
}

void trlweKeyGen(uint32_t *trlwekey, int keylen)
{
    srand((int)time(NULL));
    for (int i = 0; i < keylen; i++)
    {
        trlwekey[i] = rand() % 2;
    }
}

void trlweKeyGen64(uint64_t *trlwekey, int keylen)
{
    srand((int)time(NULL));
    for (int i = 0; i < keylen; i++)
    {
        trlwekey[i] = rand() % 2;
    }
}



void TwistFFT(int32_t *a, Params128 &p128, cuDoubleComplex *h_Comp_a)
{
    int M = p128.N / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    for (int i = 0; i < M; i++)
    {
        h_Comp_a[i].x = a[i];
        h_Comp_a[i].y = a[i + M];
        h_Comp_a[i] = cuCmul(h_Comp_a[i], p128.twist[i]);
    }
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_FORWARD);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}

void TwistFFTlong(uint64_t *a, Params128 &p128, cuDoubleComplex *h_Comp_a)
{
    // cout << "\nfft uint to int\n";
    // for (int i = 0; i < p128.nbar; i++)
    // {
    //     cout << a[i] << ", ";
    // }
    // cout << endl;
    
    int M = p128.nbar / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    for (int i = 0; i < M; i++)
    {
        h_Comp_a[i].x = a[i];
        h_Comp_a[i].y = a[i + M];
        h_Comp_a[i] = cuCmul(h_Comp_a[i], p128.twistlong[i]);
    }
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_FORWARD);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}

void TwistFFTlong(int64_t *a, Params128 &p128, cuDoubleComplex *h_Comp_a)
{
    // cout << "\nfft uint to int\n";
    // for (int i = 0; i < p128.nbar; i++)
    // {
    //     cout << a[i] << ", ";
    // }
    // cout << endl;
    
    int M = p128.nbar / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    for (int i = 0; i < M; i++)
    {
        h_Comp_a[i].x = a[i];
        h_Comp_a[i].y = a[i + M];
        h_Comp_a[i] = cuCmul(h_Comp_a[i], p128.twistlong[i]);
    }
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_FORWARD);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}


void TwistIFFT(cuDoubleComplex *h_Comp_a, Params128 &p128, double *product)
{
    int M = p128.N / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_INVERSE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++)
    {
        cuDoubleComplex twist;
        twist.x = p128.twist[i].x;
        twist.y = (-1) * p128.twist[i].y;
        // normalize
        h_Comp_a[i].x = h_Comp_a[i].x / M;
        h_Comp_a[i].y = h_Comp_a[i].y / M;
        h_Comp_a[i] = cuCmul(h_Comp_a[i], twist);
        product[i] = h_Comp_a[i].x;
        product[i + M] = h_Comp_a[i].y;
    }

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}

void TwistIFFTlong(cuDoubleComplex *h_Comp_a, Params128 &p128, double *product)
{
    int M = p128.nbar / 2;
    cuDoubleComplex *d_Comp_a;
    CHECK(cudaMalloc((void **)&d_Comp_a, M * sizeof(cuDoubleComplex)));
    CHECK(cudaMemcpy(d_Comp_a, h_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cuDoubleComplex *)d_Comp_a, (cuDoubleComplex *)d_Comp_a, CUFFT_INVERSE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_Comp_a, d_Comp_a, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++)
    {
        cuDoubleComplex twist = cuConj(p128.twistlong[i]);
        // normalize
        h_Comp_a[i].x = h_Comp_a[i].x / M;
        h_Comp_a[i].y = h_Comp_a[i].y / M;
        h_Comp_a[i] = cuCmul(h_Comp_a[i], twist);
        product[i] = h_Comp_a[i].x;
        product[i + M] = h_Comp_a[i].y;
    }

    cufftDestroy(plan);
    CHECK(cudaFree(d_Comp_a));
}


void PolyMul(uint32_t *a, uint32_t *trlwekey, uint32_t *product, Params128 &p128)
{
    int M = p128.N / 2;
    cuDoubleComplex *h_Comp_a = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_trlwekey = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    double *result = (double *)malloc(sizeof(double) * p128.N);

    TwistFFT((int32_t *)a, p128, h_Comp_a);
    TwistFFT((int32_t *)trlwekey, p128, h_Comp_trlwekey);

    for (int i = 0; i < M; i++)
    {
        h_Comp_product[i] = cuCmul(h_Comp_a[i], h_Comp_trlwekey[i]);
    }
    TwistIFFT(h_Comp_product, p128, result);
    for (int i = 0; i < p128.N; i++)
    {
        // attention!
        // (uint64_t)(round(result[i])) : double --> int , because module only support integers.
        product[i] = (uint32_t)((uint64_t)(round(result[i])) % _two32);
    }
    

    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
    free(result);
}


void PolyMullong(uint64_t *a, uint64_t *trlwekey, uint64_t *product, Params128 &p128)
{
    int M = p128.nbar / 2;
    cuDoubleComplex *h_Comp_a = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_trlwekey = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    cuDoubleComplex *h_Comp_product = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * M);
    double *result = (double *)malloc(sizeof(double) * p128.nbar);

    TwistFFTlong((int64_t *)a, p128, h_Comp_a);
    TwistFFTlong((int64_t *)trlwekey, p128, h_Comp_trlwekey);

    for (int i = 0; i < M; i++)
    {
        h_Comp_product[i] = cuCmul(h_Comp_a[i], h_Comp_trlwekey[i]);
    }
    TwistIFFTlong(h_Comp_product, p128, result);
    for (int i = 0; i < p128.nbar; i++)
    {
        product[i] = (uint64_t)(fmod(round(result[i]), _two64));   
    }
    

    free(h_Comp_a);
    free(h_Comp_trlwekey);
    free(h_Comp_product);
    free(result);
}


void trlweSymEnc(uint32_t *vecmu, uint32_t *trlwekey, double stdev, Params128 &p128, uint32_t **c)
{
    uint32_t *a = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    for (int i = 0; i < p128.N; i++)
    {
        a[i] = g();
        c[1][i] = a[i];
    }

    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(a, trlwekey, product, p128);

    uint32_t *ga = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    gaussian32(vecmu, ga, stdev, p128.N);

    for (int i = 0; i < p128.N; i++)
    {
        // attention!
        // python code:  np.array([ga - mul, a], dtype=np.uint32)
        // c[0][i] = (ga[i] - product[i]) % (_two32);
        c[0][i] = (uint32_t)(ga[i] - product[i]);
    }

    free(a);
    free(product);
    free(ga);
}


void trlweSymEnc64(uint64_t *vecmu, uint64_t *trlwekey, double stdev, Params128 &p128, uint64_t **c)
{
    uint64_t *a = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);

    for (int i = 0; i < p128.nbar; i++)
    {
        a[i] = g();    
        c[1][i] = a[i];
    }

    uint64_t *product = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    PolyMullong(a, trlwekey, product, p128);

    uint64_t *ga = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    gaussian64(vecmu, ga, stdev, p128.nbar);

    for (int i = 0; i < p128.nbar; i++)
    {
        c[0][i] = (uint64_t)(ga[i] - product[i]);
    }

    free(a);
    free(product);
    free(ga);
}

void trlweSymDec(uint32_t **c, uint32_t *trlwekey, Params128 &p128, uint32_t *mu)
{
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(c[1], trlwekey, product, p128);

    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = c[0][i] + product[i];
    }
    for (int i = 0; i < p128.N; i++)
    {
        mu[i] = Ttomu(phase[i], p128.inter);
    }

    free(product);
    free(phase);
}


void trlweSymDec64(uint64_t **c, uint64_t *trlwekey, Params128 &p128, uint64_t *mu)
{
    uint64_t *product = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    PolyMullong(c[1], trlwekey, product, p128);

    uint64_t *phase = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        phase[i] = c[0][i] + product[i];
    }
    for (int i = 0; i < p128.nbar; i++)
    {
        mu[i] = Ttomu64(phase[i], p128.interlong);
    }

    free(product);
    free(phase);
}


void Test_TRLWE()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // generate message
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
    }
    // vecmu[p128.N - 3] = mutoT(1, p128.Msize);

    // trlwekey generation
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweKeyGen(trlwekey, p128.N);

    // encryption
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(vecmu, trlwekey, p128.ks_stdev, p128, c);
    

    // decryption
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(c, trlwekey, p128, mu);
    cout << "\ntrlwe decryption result: " << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }

    free(vecmu);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
    free(mu);
}

void Test_TRLWE64()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // generate message
    uint64_t *vecmu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT64(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT64(0, p128.Msize);
        }
    }
    // vecmu[p128.nbar - 4] = mutoT64(1, p128.Msize);

    // trlwekey generation
    uint64_t *trlwekey = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trlweKeyGen64(trlwekey, p128.nbar);

    // encryption
    uint64_t **c = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    trlweSymEnc64(vecmu, trlwekey, p128.bk_stdevbar, p128, c);
    

    // decryption
    uint64_t *mu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trlweSymDec64(c, trlwekey, p128, mu);
    cout << "\nTest_TRLWE64 decryption result: " << endl;
    for (int i = 0; i < p128.nbar; i++)
    {
        // cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }

    free(vecmu);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
    free(mu);
}


void trgswSymEnc(uint32_t *vecmu, uint32_t *trlwekey, Params128 &p128, uint32_t ***c)
{
    uint32_t **muh = (uint32_t **)malloc(p128.bk_l * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l; i++)
    {
        muh[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            muh[i][j] = p128.H[i] * vecmu[j];
        }
    }

    int lines = 2 * p128.bk_l;
    uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        vec_zero[i] = 0;
    }
    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc(vec_zero, trlwekey, p128.bk_stdev, p128, c[i]); 
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_l; i < lines; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][1][j] += muh[i - p128.bk_l][j];
        }
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}


void trgswSymEnc64(uint64_t *vecmu, uint64_t *trlwekey, Params128 &p128, uint64_t ***c)
{
    uint64_t **muh = (uint64_t **)malloc(p128.bk_lbar * sizeof(uint64_t *));
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        muh[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            muh[i][j] = p128.Hbar[i] * vecmu[j];
        }
    }

    int lines = 2 * p128.bk_lbar;
    uint64_t *vec_zero = (uint64_t *)malloc(sizeof(uint64_t) * p128.nbar);
    for (int i = 0; i < p128.nbar; i++)
    {
        vec_zero[i] = 0;
    }

    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc64(vec_zero, trlwekey, p128.bk_stdevbar, p128, c[i]); 
    }

    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_lbar; i < lines; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            c[i][1][j] += muh[i - p128.bk_lbar][j];
        }
    }

    for (int i = 0; i < p128.bk_lbar; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}


void trgswSymEnc_2(uint64_t *vecmu, uint32_t *trlwekey, Params128 &p128, uint32_t*** c)
{
    uint32_t **muh = (uint32_t **)malloc(p128.bk_l * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l; i++)
    {
        muh[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            muh[i][j] = p128.H[i] * vecmu[j];
        }
    }

    int lines = 2 * p128.bk_l;
    uint32_t *vec_zero = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        vec_zero[i] = 0;
    }

    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc(vec_zero, trlwekey, p128.bk_stdev, p128, c[i]); 
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_l; i < lines; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][1][j] += muh[i - p128.bk_l][j];
        }
    }

    for (int i = 0; i < p128.bk_l; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}

void trgswSymEnc64_2(double mu, uint64_t *trlwekey, Params128 &p128, uint64_t*** c)
{
    uint64_t **muh = (uint64_t **)malloc(p128.bk_lbar * sizeof(uint64_t *));
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        muh[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 1; j < p128.nbar; j++)
        {
            muh[i][j] = 0;
        }
        muh[i][0] = DoubleToUint64(p128.Hbar[i] * mu);
    }

    int lines = 2 * p128.bk_lbar;    
    uint64_t *vec_zero = (uint64_t *)malloc(sizeof(uint64_t) * p128.nbar);
    for (int i = 0; i < p128.nbar; i++)
    {
        vec_zero[i] = 0;
    }
    for (int i = 0; i < lines; i++)
    {
        trlweSymEnc64(vec_zero, trlwekey, p128.bk_stdevbar, p128, c[i]); 
    }


    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            c[i][0][j] += muh[i][j];
        }
    }
    for (int i = p128.bk_lbar; i < lines; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            c[i][1][j] += muh[i - p128.bk_lbar][j];
        }
    }


    for (int i = 0; i < p128.bk_lbar; i++)
    {
        free(muh[i]);
    }
    free(muh);
    free(vec_zero);
}


void trgswSymDec(uint32_t ***c, uint32_t *trlwekey, Params128 &p128, uint32_t* mu)
{
    uint32_t *phase = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t *product = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    PolyMul(c[0][1], trlwekey, product, p128);
    for (int i = 0; i < p128.N; i++)
    {
        phase[i] = (c[0][0][i] + product[i]) * p128.Bg;
        mu[i] = Ttomu(phase[i], p128.inter);
    }
    free(phase);
    free(product);
}


void trgswSymDec64(uint64_t ***c, uint64_t *trlwekey, Params128 &p128, uint64_t* mu)
{
    uint64_t *phase = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    uint64_t *product = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    PolyMullong(c[0][1], trlwekey, product, p128);
    for (int i = 0; i < p128.nbar; i++)
    {
        phase[i] = (c[0][0][i] + product[i]) * p128.Bgbar;
        mu[i] = Ttomu64(phase[i], p128.interlong);
    }
    free(phase);
    free(product);
}


void Test_TRGSW()
{   
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);

    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT(0, p128.Msize);
        }
    }
    // for (int i = p128.N / 2; i < p128.N; i++)
    // {
    //     vecmu[i] = mutoT(1, p128.Msize);
    // }
    // vecmu[p128.N-5] = mutoT(0, p128.Msize);
    

    // trlwekey generation
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweKeyGen(trlwekey, p128.N);

    
    int lines = 2 * p128.bk_l;
    uint32_t ***c;
    c = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        c[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }
    trgswSymEnc(vecmu, trlwekey, p128, c);


    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trgswSymDec(c, trlwekey, p128, mu);
    cout << "\ntrgsw decryption result:" << endl;
    for (int i = 0; i < p128.N; i++)
    {
        cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }
    

    free(vecmu);
    free(trlwekey);
    free(mu);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(c[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(c[i]);
    }
    free(c);
}

void Test_TRGSW64()
{   
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // Params128 p128 = Params128(1024, 630, 2, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);


    uint64_t *vecmu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        if (i % 2 == 1)
        {
            vecmu[i] = mutoT64(0, p128.Msize);
        }
        else
        {
            vecmu[i] = mutoT64(1, p128.Msize);
        }
    }
    // vecmu[p128.nbar - 5] = mutoT64(1, p128.Msize);
    

    // trlwekey generation
    uint64_t *trlwekey = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trlweKeyGen64(trlwekey, p128.nbar);

    
    int lines = 2 * p128.bk_lbar;
    uint64_t ***c;
    c = (uint64_t ***)malloc(lines * sizeof(uint64_t **));

    for (int i = 0; i < lines; i++)
    {
        c[i] = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            c[i][j] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
        }
    }
    trgswSymEnc64(vecmu, trlwekey, p128, c);


    uint64_t *mu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trgswSymDec64(c, trlwekey, p128, mu);
    cout << "\ntrgsw decryption result:" << endl;
    for (int i = 0; i < p128.nbar; i++)
    {
        // cout << mu[i] << " ";
        if(i % 2 == 1 && mu[i] != 0)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && mu[i] != 1)
        {
            cout << "\ni:" << i << " get wrong!" << endl;
        }
    }
    

    free(vecmu);
    free(trlwekey);
    free(mu);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(c[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(c[i]);
    }
    free(c);
}


void trgswfftSymEnc(uint64_t *mu, uint32_t *trlwekey, Params128 &p128, cuDoubleComplex ***trgswfftcipher)
{
    int lines = 2 * p128.bk_l;
    uint32_t ***trgsw;
    trgsw = (uint32_t ***)malloc(lines * sizeof(uint32_t **));

    for (int i = 0; i < lines; i++)
    {
        trgsw[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgsw[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }
    trgswSymEnc_2(mu, trlwekey, p128, trgsw);
    cout << "\ntrgswcipher without fft\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                cout << trgsw[i][j][k] << ", ";
            }   

        }
        cout << endl;
    }


    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            // attention!
            TwistFFT((int32_t *)trgsw[i][j], p128, trgswfftcipher[i][j]);
        }   
    }

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgsw[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgsw[i]);
    }
    free(trgsw);
}

void trgswfftSymEnc64(double mu, uint64_t *trlwekey, Params128 &p128, cuDoubleComplex ***trgswfftcipher)
{
    int lines = 2 * p128.bk_lbar;
    uint64_t ***trgsw;
    trgsw = (uint64_t ***)malloc(lines * sizeof(uint64_t **));
    for (int i = 0; i < lines; i++)
    {
        trgsw[i] = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgsw[i][j] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
        }
    }
    trgswSymEnc64_2(mu, trlwekey, p128, trgsw);

    int64_t *TRGSW = (int64_t *)malloc(p128.nbar * sizeof(int64_t));
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            TwistFFTlong(trgsw[i][j], p128, trgswfftcipher[i][j]);
        }   
    }

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgsw[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgsw[i]);
    }
    free(trgsw);
    free(TRGSW);
}



__global__ void externalproduct(cudaPitchedPtr devPitchedPtr, cudaPitchedPtr devPitchedPtr_2, cudaPitchedPtr devPitchedPtr_3, cudaExtent extent)
{
    cuDoubleComplex *devPtr = (cuDoubleComplex *)devPitchedPtr.ptr;
    cuDoubleComplex *sliceHead, *rowHead;
    cuDoubleComplex *devPtr_2 = (cuDoubleComplex *)devPitchedPtr_2.ptr;
    cuDoubleComplex *sliceHead_2, *rowHead_2;
    cuDoubleComplex *devPtr_3 = (cuDoubleComplex *)devPitchedPtr_3.ptr;
    cuDoubleComplex *sliceHead_3, *rowHead_3;

    for (int z = 0; z < extent.depth; z++)
    {
        sliceHead = (cuDoubleComplex *)((char *)devPtr + z * devPitchedPtr.pitch * extent.height);
        sliceHead_2 = (cuDoubleComplex *)((char *)devPtr_2 + z * devPitchedPtr_2.pitch * extent.height);
        sliceHead_3 = (cuDoubleComplex *)((char *)devPtr_3 + z * devPitchedPtr_3.pitch * extent.height);
        for (int y = 0; y < extent.height; y++)
        {
            rowHead = (cuDoubleComplex*)((char *)sliceHead + y * devPitchedPtr.pitch);
            rowHead_2 = (cuDoubleComplex*)((char *)sliceHead_2 + y * devPitchedPtr_2.pitch);
            rowHead_3 = (cuDoubleComplex*)((char *)sliceHead_3 + y * devPitchedPtr_3.pitch);
            for (int x = 0; x < extent.width / sizeof(cuDoubleComplex); x++)
            {
                rowHead_3[x] = cuCmul(rowHead[x], rowHead_2[x]);
            }
        }
    }

}

void ExternalProduct(cuDoubleComplex ***trgswfftcipher, uint32_t **trlwecipher, Params128 &p128, uint32_t **res)
{
    int lines = 2 * p128.bk_l;
    int M = p128.N / 2;

    // t = np.uint32((trlwecipher + p128.offset)%2**32)
    // t size: 2 * p128.N      ok!
    uint32_t **t = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            // trlwecipher + offset may surpass uint32_t ??
            t[i][j] = (uint32_t)((trlwecipher[i][j] + (uint64_t)p128.offset) % _two32);
        }
    }

    // t1 = np.array([t >> i for i in p128.decbit]) 
    // t1 size: p128.bk_l * 2 * p128.N        ok!
    uint32_t ***t1;
    t1 = (uint32_t ***)malloc(p128.bk_l * sizeof(uint32_t **));
    for (int i = 0; i < p128.bk_l; i++)
    {
        t1[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t1[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }


    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t[j][k] >> p128.decbit[i];
            }
        }
    }

    // t2=t1&(p128.Bg - 1)
    // t2 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bg - 1);
            }
        }
    }

    // t3=t2-p128.Bg // 2
    // t3 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] - p128.Bg / 2;
            }   
        }
    }


    // decvec = np.concatenate([t3[:, 0], t3[:, 1]])
    // decvec size: (p128.bk_l * 2) * p128.N        ok!
    uint32_t **decvec = (uint32_t **)malloc(p128.bk_l * 2 * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        decvec[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < p128.bk_l; i++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                decvec[i+j*2][k] = t1[i][j][k];
            }
        }   
    }


    // decvecfft = TwistFFT(np.int32(decvec), p128.twist, dim=2)
    // decvecfft size: (p128.bk_l * 2) * (p128.N / 2)        ok!
    cuDoubleComplex **decvecfft = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex)*2*p128.bk_l);
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        decvecfft[i] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.N / 2);
    }
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        TwistFFT((int32_t *)decvec[i], p128, decvecfft[i]);
    }

    // cout << setiosflags(ios::scientific) << setprecision(8);
    // cout << "\ndecvecfft\n";
    // for (int i = 0; i < 2*p128.bk_l; i++)
    // {
    //     for (int j = 0; j < p128.N / 2; j++)
    //     {
    //         cout << decvecfft[i][j].x << "+" << decvecfft[i][j].y << "j, ";
    //     }    
    //     cout << endl;
    // }
    
    
    
    // t4 = decvecfft.reshape(2 * p128.bk_l, 1, p128.N // 2) * trgswfftcipher
    // t4 size: (2 * p128.bk_l) * 2 * (p128.N / 2)     ok!
    cuDoubleComplex ***t4;
    t4 = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        t4[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t4[i][j] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
        }
    }

    size_t width = p128.N / 2;
    size_t height = 2;
    size_t depth = 2 * p128.bk_l;
    cuDoubleComplex *h_trgswfftcipher, *h_decvecfft, *h_result;
    cudaPitchedPtr d_trgswfftcipher, d_decvecfft, d_result;
    cudaExtent extent;
    cudaMemcpy3DParms cpyParm;


    h_trgswfftcipher = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    h_result = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                h_trgswfftcipher[i*height*width+j*width+k] = trgswfftcipher[i][j][k];
            }
        }   
    }
    // process trgswfftcipher
    // alloc device memory
    extent = make_cudaExtent(sizeof(cuDoubleComplex) * width, height, depth);
    cudaMalloc3D(&d_trgswfftcipher, extent);

    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_trgswfftcipher, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.dstPtr = d_trgswfftcipher;
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);


    // process decvecfft
    h_decvecfft = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * width * height * depth);
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                h_decvecfft[i*height*width+j*width+k] = decvecfft[i][k];
            }
        }   
    }
    
    cudaMalloc3D(&d_decvecfft, extent);
    cpyParm = {0};
    cpyParm.srcPtr = make_cudaPitchedPtr((void*)h_decvecfft, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.dstPtr = d_decvecfft;
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyParm);

    cudaMalloc3D(&d_result, extent);

    // call kernel 
    externalproduct<<<1,1>>>(d_trgswfftcipher, d_decvecfft, d_result, extent);
    cudaDeviceSynchronize();
    cpyParm = { 0 };
    cpyParm.srcPtr = d_result;
    cpyParm.dstPtr = make_cudaPitchedPtr((void*)h_result, sizeof(cuDoubleComplex) * width, width, height);
    cpyParm.extent = extent;
    cpyParm.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&cpyParm);


    // cout << "\nt4\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = h_result[i*width*height+j*width+k];
                // cout << t4[i][j][k].x << "+" << t4[i][j][k].y << "j, ";
            }
        }
        // cout << endl;
    }
    // cout << endl;



    
    // t5 = t4.sum(axis=0)
    // t5 size: 2 * (p128.N / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }

    // t6 = TwistIFFT(t5, p128.twist, axis=1)
    // t6 size : 2 * p128.N
    double **t6 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t6[i] = (double *)malloc(p128.N * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        TwistIFFT(t5[i], p128, t6[i]);
    }

    // res=np.array(t6, dtype=np.uint32)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            res[i][j] = (uint32_t)t6[i][j];
        }
    }
    
    

    for (int i = 0; i < 2; i++)
    {
        free(t[i]);
    }
    free(t);
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t1[i][j]);
        }
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        free(t1[i]);
    }
    free(t1);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvec[i]);
    }
    free(decvec);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvecfft[i]);
    }
    free(decvecfft);
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t4[i][j]);
        }
    }
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        free(t4[i]);
    }
    free(t4);
    free(h_trgswfftcipher);
    free(h_decvecfft);
    free(h_result);
    CHECK(cudaFree(d_trgswfftcipher.ptr));
    CHECK(cudaFree(d_decvecfft.ptr));
    CHECK(cudaFree(d_result.ptr));
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
    for (int i = 0; i < 2; i++)
    {
        free(t6[i]);
    }
    free(t6);
}


void trgswfftExternalProduct(cuDoubleComplex ***trgswfftcipher, uint32_t **trlwecipher, Params128 &p128, uint32_t **res)
{
    int lines = 2 * p128.bk_l;
    int M = p128.N / 2;

    // t = np.uint32((trlwecipher + p128.offset)%2**32)
    // t size: 2 * p128.N      ok!
    uint32_t **t = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            // trlwecipher + offset may surpass uint32_t ??
            t[i][j] = (uint32_t)((trlwecipher[i][j] + (uint64_t)p128.offset) % _two32);
        }
    }

    // t1 = np.array([t >> i for i in p128.decbit]) 
    // t1 size: p128.bk_l * 2 * p128.N        ok!
    uint32_t ***t1;
    t1 = (uint32_t ***)malloc(p128.bk_l * sizeof(uint32_t **));
    for (int i = 0; i < p128.bk_l; i++)
    {
        t1[i] = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t1[i][j] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
        }
    }


    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t[j][k] >> p128.decbit[i];
            }
        }
    }

    // t2=t1&(p128.Bg - 1)
    // t2 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bg - 1);
            }
        }
    }

    // t3=t2-p128.Bg // 2
    // t3 size: p128.bk_l * 2 * p128.N        ok!
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                t1[i][j][k] = t1[i][j][k] - p128.Bg / 2;
            }   
        }
    }


    // decvec = np.concatenate([t3[:, 0], t3[:, 1]])
    // decvec size: (p128.bk_l * 2) * p128.N        ok!
    uint32_t **decvec = (uint32_t **)malloc(p128.bk_l * 2 * sizeof(uint32_t *));
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        decvec[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < p128.bk_l; i++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                decvec[i+j*2][k] = t1[i][j][k];
            }
        }   
    }


    // decvecfft = TwistFFT(np.int32(decvec), p128.twist, dim=2)
    // decvecfft size: (p128.bk_l * 2) * (p128.N / 2)        ok!
    cuDoubleComplex **decvecfft = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex)*2*p128.bk_l);
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        decvecfft[i] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.N / 2);
    }
    for (int i = 0; i < 2*p128.bk_l; i++)
    {
        TwistFFT((int32_t *)decvec[i], p128, decvecfft[i]);
    }
     
    
    // t4 = decvecfft.reshape(2 * p128.bk_l, 1, p128.N // 2) * trgswfftcipher
    // t4 size: (2 * p128.bk_l) * 2 * (p128.N / 2)     ok!
    cuDoubleComplex ***t4;
    t4 = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        t4[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t4[i][j] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
        }
    }

    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = cuCmul(decvecfft[i][k], trgswfftcipher[i][j][k]);
            } 
        }
    }

    // t5 = t4.sum(axis=0)
    // t5 size: 2 * (p128.N / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }

    // t6 = TwistIFFT(t5, p128.twist, axis=1)
    // t6 size : 2 * p128.N
    double **t6 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t6[i] = (double *)malloc(p128.N * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        TwistIFFT(t5[i], p128, t6[i]);
    }

    // res=np.array(t6, dtype=np.uint32)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            res[i][j] = (uint32_t)t6[i][j];
        }
    }
    
    

    for (int i = 0; i < 2; i++)
    {
        free(t[i]);
    }
    free(t);
    for (int i = 0; i < p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t1[i][j]);
        }
    }
    for (int i = 0; i < p128.bk_l; i++)
    {
        free(t1[i]);
    }
    free(t1);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvec[i]);
    }
    free(decvec);
    for (int i = 0; i < p128.bk_l * 2; i++)
    {
        free(decvecfft[i]);
    }
    free(decvecfft);
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t4[i][j]);
        }
    }
    for (int i = 0; i < 2 * p128.bk_l; i++)
    {
        free(t4[i]);
    }
    free(t4);
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
    for (int i = 0; i < 2; i++)
    {
        free(t6[i]);
    }
    free(t6);
}

void trgswfftExternalProduct64(cuDoubleComplex ***trgswfftcipher, uint64_t **trlwecipher, Params128 &p128, uint64_t **res)
{
    cout << setiosflags(ios::scientific) << setprecision(8);
    int lines = 2 * p128.bk_lbar;
    int M = p128.nbar / 2;

    // t = [trlwecipher[0], trlwecipher[1]] + p128.offsetbar
    // t size: (2, p128.nbar)
    uint64_t **t = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            t[i][j] = trlwecipher[i][j] + p128.offsetbar;
        }
    }

    // t=np.array([t >> i for i in p128.decbitbar])
    // t1 size: (p128.bk_lbar, 2, p128.nbar)        ok!
    uint64_t ***t1;
    t1 = (uint64_t ***)malloc(p128.bk_lbar * sizeof(uint64_t **));
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        t1[i] = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t1[i][j] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
        }
    }

    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t1[i][j][k] = t[j][k] >> p128.decbitbar[i];
            }
        }
    }

    // t &= p128.Bgbar - 1
    // t2 size: (p128.bk_lbar, 2, p128.nbar)       ok!
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bgbar - 1);
            }
        }
    }

    // t -= p128.Bgbar // 2
    // t3 size: (p128.bk_lbar, 2, p128.nbar)        ok!
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t1[i][j][k] = t1[i][j][k] - p128.Bgbar / 2;
            }   
        }
    }



    // decvec = np.concatenate([t[:, 0], t[:, 1]])
    // decvec size: (p128.bk_lbar * 2, p128.nbar)        ok!
    uint64_t **decvec = (uint64_t **)malloc(p128.bk_lbar * 2 * sizeof(uint64_t *));
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        decvec[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    int cnt = 0;
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < p128.bk_lbar; i++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                decvec[cnt][k] = t1[i][j][k];
            }
            cnt++;
        }   
    }


    // decvecfft = TwistFFTlong(np.int64(decvec), p128.twistlong, dim=2)
    // decvecfft size: (p128.bk_lbar * 2, p128.nbar / 2)        ok!
    cuDoubleComplex **decvecfft = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex)*2*p128.bk_lbar);
    for (int i = 0; i < 2*p128.bk_lbar; i++)
    {
        decvecfft[i] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.nbar / 2);
    }
    for (int i = 0; i < 2*p128.bk_lbar; i++)
    {
        TwistFFTlong((int64_t *)decvec[i], p128, decvecfft[i]);
        // TwistFFTlong(decvec[i], p128, decvecfft[i]);
    }
     
    
    // t = decvecfft.reshape(2 * p128.bk_lbar, 1, p128.nbar // 2) * trgswfftcipher
    // t4 size: (2 * p128.bk_lbar, 2, p128.nbar / 2)     ok!
    cuDoubleComplex ***t4;
    t4 = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        t4[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t4[i][j] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
        }
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = cuCmul(decvecfft[i][k], trgswfftcipher[i][j][k]);
            } 
        }
    }

    // t = t.sum(axis=0)
    // t5 size: (2, p128.nbar / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }


    // t = TwistIFFTlong(t, p128.twistlong, axis=1)
    // t6 size : (2, p128.nbar)
    double **t6 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t6[i] = (double *)malloc(p128.nbar * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        TwistIFFTlong(t5[i], p128, t6[i]);
    }
    

    // return np.array(t%(float(2)**64), dtype=np.uint64)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            // res[i][j] = (uint64_t)(abs(fmod(t6[i][j], _two64)));
            double aa = fmod(t6[i][j], _two64);
            if(sign(aa) == -1)
            {
                res[i][j] = (uint64_t)(aa + _two64);
            }
            else{
                res[i][j] = (uint64_t)aa;
            }
            // cout << res[i][j] << ", ";
        }
        // cout << endl;
    }
    // cout << endl;
    
    

    for (int i = 0; i < 2; i++)
    {
        free(t[i]);
    }
    free(t);
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t1[i][j]);
        }
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        free(t1[i]);
    }
    free(t1);
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        free(decvec[i]);
    }
    free(decvec);
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        free(decvecfft[i]);
    }
    free(decvecfft);
    for (int i = 0; i < 2 * p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t4[i][j]);
        }
    }
    for (int i = 0; i < 2 * p128.bk_lbar; i++)
    {
        free(t4[i]);
    }
    free(t4);
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
    for (int i = 0; i < 2; i++)
    {
        free(t6[i]);
    }
    free(t6);
}

void trgswfftExternalProduct64(cuDoubleComplex ***trgswfftcipher, double **trlwecipher, Params128 &p128, uint64_t **res)
{
    cout << setiosflags(ios::scientific) << setprecision(8);
    int lines = 2 * p128.bk_lbar;
    int M = p128.nbar / 2;

    // t = [trlwecipher[0], trlwecipher[1]] + p128.offsetbar
    // t size: (2, p128.nbar)
    cout << "\nt\n";
    double **t = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t[i] = (double *)malloc(p128.nbar * sizeof(double));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            t[i][j] = trlwecipher[i][j] + p128.offsetbar;
            cout << t[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    // t=np.array([t >> i for i in p128.decbitbar])
    // t1 size: (p128.bk_lbar, 2, p128.nbar)        ok!
    int64_t ***t1;
    t1 = (int64_t ***)malloc(p128.bk_lbar * sizeof(int64_t **));
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        t1[i] = (int64_t **)malloc(2 * sizeof(int64_t *));
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t1[i][j] = (int64_t *)malloc(p128.nbar * sizeof(int64_t));
        }
    }

    cout << "\nt1\n";
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t1[i][j][k] = t[j][k] / pow(2, p128.decbitbar[i]);
                if(fmod(t[j][k], pow(2, p128.decbitbar[i])) < 0){
                    t1[i][j][k] -= 1;
                }
                cout << t1[i][j][k] << ", ";
            }
        }
        cout << endl;
    }
    cout << endl;

    // t &= p128.Bgbar - 1
    // t2 size: (p128.bk_lbar, 2, p128.nbar)       ok!
    cout << "\nt2\n";
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t1[i][j][k] = t1[i][j][k] & (p128.Bgbar - 1);
                cout << t1[i][j][k] << ", ";
            }
        }
        cout << endl;
    }
    cout << endl;

    // t -= p128.Bgbar // 2
    // t3 size: (p128.bk_lbar, 2, p128.nbar)        ok!
    cout << "\nt3\n";
    int64_t ***t3;
    t3 = (int64_t ***)malloc(p128.bk_lbar * sizeof(int64_t **));
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        t3[i] = (int64_t **)malloc(2 * sizeof(int64_t *));
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t3[i][j] = (int64_t *)malloc(p128.nbar * sizeof(int64_t));
        }
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                t3[i][j][k] = t1[i][j][k] - p128.Bgbar / 2;
                cout << t3[i][j][k] << ", ";
            }   
        }
        cout << endl;
    }
    cout << endl;



    // decvec = np.concatenate([t[:, 0], t[:, 1]])
    // decvec size: (p128.bk_lbar * 2, p128.nbar)        ok!
    int64_t **decvec = (int64_t **)malloc(p128.bk_lbar * 2 * sizeof(int64_t *));
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        decvec[i] = (int64_t *)malloc(p128.nbar * sizeof(int64_t));
    }
    int cnt = 0;
    cout << "\ndecvec\n";
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < p128.bk_lbar; i++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                decvec[cnt][k] = t3[i][j][k];
                cout << decvec[cnt][k] << ", ";
            }
            cnt++;
            cout << endl;
        }   
        cout << endl;
    }
    cout << endl;


    // decvecfft = TwistFFTlong(np.int64(decvec), p128.twistlong, dim=2)
    // decvecfft size: (p128.bk_lbar * 2, p128.nbar / 2)        ok!
    cuDoubleComplex **decvecfft = (cuDoubleComplex **)malloc(sizeof(cuDoubleComplex)*2*p128.bk_lbar);
    for (int i = 0; i < 2*p128.bk_lbar; i++)
    {
        decvecfft[i] = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * p128.nbar / 2);
    }
    for (int i = 0; i < 2*p128.bk_lbar; i++)
    {
        TwistFFTlong(decvec[i], p128, decvecfft[i]);
        // TwistFFTlong(decvec[i], p128, decvecfft[i]);
    }

    cout << "\ndecvecfft\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < M; j++)
        {
            cout << decvecfft[i][j].x << "+" << decvecfft[i][j].y << "j, ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "\ntrgswfftcipher\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                cout << trgswfftcipher[i][j][k].x << "+" << trgswfftcipher[i][j][k].y << "j, ";
            } 
        }
        cout << endl;
    }
    cout << endl;


     
    
    // t = decvecfft.reshape(2 * p128.bk_lbar, 1, p128.nbar // 2) * trgswfftcipher
    // t4 size: (2 * p128.bk_lbar, 2, p128.nbar / 2)     ok!
    cuDoubleComplex ***t4;
    t4 = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        t4[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t4[i][j] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
        }
    }
    cout << "\nt4\n";
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[i][j][k] = cuCmul(decvecfft[i][k], trgswfftcipher[i][j][k]);
                cout << t4[i][j][k].x << "+" << t4[i][j][k].y << "j, ";
            } 
        }
        cout << endl;
    }
    cout << endl;

    // t = t.sum(axis=0)
    // t5 size: (2, p128.nbar / 2)    ok!
    cuDoubleComplex **t5 = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    for (int i = 0; i < 2; i++)
    {
        t5[i] = (cuDoubleComplex *)malloc(M * sizeof(cuDoubleComplex));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[i][j].x = 0;
            t5[i][j].y = 0;
            for (int k = 0; k < lines; k++)
            {
                t5[i][j] = cuCadd(t5[i][j], t4[k][i][j]);
            }
        }
    }

    cout << "\nt5\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar / 2; j++)
        {
            cout << t5[i][j].x << "+" << t5[i][j].y << "j, ";
        }   
        cout << endl;
    }
    cout << endl;
    


    // t = TwistIFFTlong(t, p128.twistlong, axis=1)
    // t6 size : (2, p128.nbar)
    double **t6 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        t6[i] = (double *)malloc(p128.nbar * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        TwistIFFTlong(t5[i], p128, t6[i]);
    }
    
    cout << "\nt6\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cout << t6[i][j] << ", ";
        }   
        cout << endl;
    }
    cout << endl;

    // return np.array(t%(float(2)**64), dtype=np.uint64)
    cout << "\ncprod\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            res[i][j] = DoubleToUint64(t6[i][j]);
            // double aa = fmod(t6[i][j], _two64);
            // if(sign(aa) == -1)
            // {
            //     res[i][j] = (uint64_t)(aa + _two64);
            // }
            // else{
            //     res[i][j] = (uint64_t)aa;
            // }
            cout << res[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
    
    

    for (int i = 0; i < 2; i++)
    {
        free(t[i]);
    }
    free(t);
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t1[i][j]);
        }
    }
    for (int i = 0; i < p128.bk_lbar; i++)
    {
        free(t1[i]);
    }
    free(t1);
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        free(decvec[i]);
    }
    free(decvec);
    for (int i = 0; i < p128.bk_lbar * 2; i++)
    {
        free(decvecfft[i]);
    }
    free(decvecfft);
    for (int i = 0; i < 2 * p128.bk_lbar; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(t4[i][j]);
        }
    }
    for (int i = 0; i < 2 * p128.bk_lbar; i++)
    {
        free(t4[i]);
    }
    free(t4);
    for (int i = 0; i < 2; i++)
    {
        free(t5[i]);
    }
    free(t5);
    for (int i = 0; i < 2; i++)
    {
        free(t6[i]);
    }
    free(t6);
}



void Test_ExternalProduct()
{
    Params128 p128 = Params128(2, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);

    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(0, p128.Msize);
        }
        else
        {
            mu[i] = mutoT(1, p128.Msize);
        }
    }
    // mu[1] = 0;
    // mu[p128.N-2] = mutoT(1, p128.Msize);
    // mu[p128.N-3]=mutoT(0, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
    }
    pgsw[0] = pow(2, 32);

    // trlwekey generation
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweKeyGen(trlwekey, p128.N);
    cout << "\ntrlwekey\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << trlwekey[i] << ", ";
    }
    cout << endl;
    

    // encryption
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));

    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc(pgsw, trlwekey, p128, trgswfftcipher);

    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint32_t **cprod = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    ExternalProduct(trgswfftcipher, trlwecipher, p128, cprod);
    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cprod, trlwekey, p128, msg);

    cout << "\n\nexternal product decryption result:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
        if(i % 2 == 1 && msg[i] != 1)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && msg[i] != 0)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
    }
    


    free(mu);
    free(pgsw);
    free(trlwekey);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}

void Test_ExternalProduct64()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // Params128 p128 = Params128(1024, 630, 2, 2, 10, 4, 9, 8, 2, 10, 3, 0, 0, 0, 0, 2);

    // generate message
    uint64_t *mu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT64(0, p128.Msize);
        }
        else
        {
            mu[i] = mutoT64(1, p128.Msize);
        }
    }
    mu[1] = 0;
    // mu[p128.nbar-2] = mutoT64(1, p128.Msize);
    // mu[p128.nbar-3]=mutoT64(0, p128.Msize);
    double pgsw = _two64;

    // trlwekey generation
    uint64_t *trlwekey = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trlweKeyGen64(trlwekey, p128.nbar);
    

    // encryption
    int lines = 2 * p128.bk_lbar;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));
    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.nbar / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc64(pgsw, trlwekey, p128, trgswfftcipher);

    uint64_t **trlwecipher = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    trlweSymEnc64(mu, trlwekey, p128.bk_stdevbar, p128, trlwecipher);

    

    uint64_t **cprod = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    trgswfftExternalProduct64(trgswfftcipher, trlwecipher, p128, cprod);

    
    uint64_t *msg = (uint64_t *)malloc(sizeof(uint64_t) * p128.nbar);
    trlweSymDec64(cprod, trlwekey, p128, msg);

    cout << "\nTest_ExternalProduct64 decryption result:\n";
    for (int i = 0; i < p128.nbar; i++)
    {
        cout << msg[i] << " ";
        if(i % 2 == 1 && msg[i] != 1)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
        if(i % 2 == 0 && msg[i] != 0)
        {
            cout << "\ni: " << i << " get wrong!" << endl;
        }
    }
    


    free(mu);
    free(trlwekey);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}

void CMUXFFT(cuDoubleComplex ***CFFT, uint32_t **d1, uint32_t **d0, uint32_t **cprod, Params128 &p128)
{
    // return ExternalProduct(CFFT, d1 - d0, p128) + d0
    uint32_t **tmp = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
        }
    }
    
    ExternalProduct(CFFT, tmp, p128, cprod);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cprod[i][j] = cprod[i][j] + d0[i][j];
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
}


void CMUXFFT64(cuDoubleComplex ***CFFT, uint64_t **d1, uint64_t **d0, uint64_t **cprod, Params128 &p128)
{
    // return ExternalProduct(CFFT, d1 - d0, p128) + d0
    uint64_t **tmp = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
        }
    }
    
    trgswfftExternalProduct64(CFFT, tmp, p128, cprod);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cprod[i][j] = cprod[i][j] + d0[i][j];
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
}


void CMUXFFT64(cuDoubleComplex ***CFFT, double **d1, double **d0, uint64_t **cprod, Params128 &p128)
{
    // return ExternalProduct(CFFT, d1 - d0, p128) + d0
    double **tmp = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (double *)malloc(p128.nbar * sizeof(double));
    }
    cout << "\nd1-d0\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
            cout << tmp[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    double **cprod0 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        cprod0[i] = (double *)malloc(p128.nbar * sizeof(double));
    }
    
    trgswfftExternalProduct64(CFFT, tmp, p128, cprod);

    cout << "\nd0\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cout << d0[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cprod[i][j] = DoubleToUint64(cprod[i][j] + d0[i][j]);
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
    for (int i = 0; i < 2; i++)
    {
        free(cprod0[i]);
    }
    free(cprod0);
}

void CMUXFFT64(cuDoubleComplex ***CFFT, uint64_t **d1, double **d0, uint64_t **cprod, Params128 &p128)
{
    // return ExternalProduct(CFFT, d1 - d0, p128) + d0
    double **tmp = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (double *)malloc(p128.nbar * sizeof(double));
    }
    cout << "\nd1-d0\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
            cout << tmp[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;

    double **cprod0 = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        cprod0[i] = (double *)malloc(p128.nbar * sizeof(double));
    }
    
    trgswfftExternalProduct64(CFFT, tmp, p128, cprod);

    cout << "\nd0\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cout << d0[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cprod[i][j] = DoubleToUint64(cprod[i][j] + d0[i][j]);
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
    for (int i = 0; i < 2; i++)
    {
        free(cprod0[i]);
    }
    free(cprod0);
}



void CMUXFFT(cuDoubleComplex ***CFFT, int64_t **d1, int64_t **d0, int64_t **cprod, Params128 &p128)
{
    // return ExternalProduct(CFFT, d1 - d0, p128) + d0
    uint32_t **tmp = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            tmp[i][j] = d1[i][j] - d0[i][j];
        }
    }

    uint32_t **cprod0 = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod0[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    
    // ExternalProduct(CFFT, tmp, p128, cprod0);
    trgswfftExternalProduct(CFFT, tmp, p128, cprod0);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cprod[i][j] = cprod0[i][j] + d0[i][j];
        }
    }

    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
    for (int i = 0; i < 2; i++)
    {
        free(cprod0[i]);
    }
    free(cprod0);
}

void Test_CMUXFFT()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
    // generate message
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(1, p128.Msize);
        }
        else
        {
            mu[i] = mutoT(0, p128.Msize);
        }
    }
    uint32_t *mu2 = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu2[i] = 0;
    }
    mu2[2] = mutoT(4, p128.Msize);
    mu2[1] = mutoT(5, p128.Msize);
    uint64_t *pgsw = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        pgsw[i] = 0;
    }
    pgsw[0] = pow(2, 32);
    

    // trlwekey generation
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    trlweKeyGen(trlwekey, p128.N);


    // encryption
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));
    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc(pgsw, trlwekey, p128, trgswfftcipher);


    uint32_t **trlwecipher = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint32_t **trlwecipher2 = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher2[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu2, trlwekey, p128.ks_stdev, p128, trlwecipher2);

    uint32_t **cprod = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    // when trgswfftcipher == 0, choose trlwecipher, otherwise trlwecipher2
    CMUXFFT(trgswfftcipher, trlwecipher2, trlwecipher, cprod, p128);
    
    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(cprod, trlwekey, p128, msg);

    cout << "\n\nTest CMUXFFT result:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }


    free(mu);
    free(mu2);
    free(pgsw);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher2[i]);
    }
    free(trlwecipher2);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}


void Test_CMUXFFT64()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
    // generate message
    uint64_t *mu = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT64(1, p128.Msize);
        }
        else
        {
            mu[i] = mutoT64(0, p128.Msize);
        }
    }
    uint64_t *mu2 = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        mu2[i] = 0;
    }
    mu2[2] = mutoT64(1, p128.Msize);
    mu2[1] = mutoT64(1, p128.Msize);
    double pgsw = _two64;
    
    // trlwekey generation
    uint64_t *trlwekey = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    trlweKeyGen64(trlwekey, p128.nbar);


    // encryption
    int lines = 2 * p128.bk_lbar;
    cuDoubleComplex ***trgswfftcipher;
    trgswfftcipher = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));
    for (int i = 0; i < lines; i++)
    {
        trgswfftcipher[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgswfftcipher[i][j] = (cuDoubleComplex *)malloc(p128.nbar / 2 * sizeof(cuDoubleComplex));
        }
    }
    trgswfftSymEnc64(pgsw, trlwekey, p128, trgswfftcipher);


    uint64_t **trlwecipher = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    trlweSymEnc64(mu, trlwekey, p128.ks_stdev, p128, trlwecipher);
    

    uint64_t **trlwecipher2 = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher2[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }
    trlweSymEnc64(mu2, trlwekey, p128.ks_stdev, p128, trlwecipher2);

    uint64_t **cprod = (uint64_t **)malloc(2 * sizeof(uint64_t *));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint64_t *)malloc(p128.nbar * sizeof(uint64_t));
    }

    // when trgswfftcipher == 0, choose trlwecipher, otherwise trlwecipher2
    CMUXFFT64(trgswfftcipher, trlwecipher2, trlwecipher, cprod, p128);
    
    uint64_t *msg = (uint64_t *)malloc(sizeof(uint64_t) * p128.nbar);
    trlweSymDec64(cprod, trlwekey, p128, msg);

    cout << "\n\nTest CMUXFFT64 result:\n";
    for (int i = 0; i < p128.nbar; i++)
    {
        cout << msg[i] << " ";
    }


    free(mu);
    free(mu2);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher2[i]);
    }
    free(trlwecipher2);
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            free(trgswfftcipher[i][j]);
        }
    }
    for (int i = 0; i < lines; i++)
    {
        free(trgswfftcipher[i]);
    }
    free(trgswfftcipher);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
    free(msg);
}


int GetNum(uint32_t *num_bits, int len)
{
    int num = 0;
    int i = 0;
    while (i < len)
    {
        num = num + num_bits[i] * pow(2, i);
        i++;
    }
    return num;
}

void GetBits(int num, uint32_t *num_bits, int len)
{
    for (int i = 0; i < len; i++)
    {
        num_bits[i] = 0;
    }
    
    int j = 0;
    while (num)
    {
        num_bits[j] = num % 2;
        num = num / 2;
        j++;
    }
}


void Database(Params128 &p128, uint32_t ***database, int flen, uint32_t *trlwekey)
{
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    // trlwekey generation
    trlweKeyGen(trlwekey, p128.N);

    for (int i = 0; i < flen; i++)
    {
        // generate message       
        GetBits(i, vecmu, p128.N);
        for (int j = 0; j < p128.N; j++)
        {
            vecmu[j] = mutoT(vecmu[j], p128.Msize);
        }

        // encryption
        trlweSymEnc(vecmu, trlwekey, p128.ks_stdev, p128, c);
        
        // attention !
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                database[i][j][k] = c[j][k];
            }
        }

        // trlweSymDec(c, trlwekey, p128, vecmu);
        // cout << GetNum(vecmu, p128.N) << endl;
    }

    free(vecmu);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
}



void DB_generation(Params128 &p128, uint32_t *database, int flen, uint32_t *trlwekey)
{
    uint32_t *vecmu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    uint32_t **c = (uint32_t **)malloc(2 * sizeof(uint32_t *));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    }

    // trlwekey generation
    trlweKeyGen(trlwekey, p128.N);

    for (int i = 0; i < flen; i++)
    {
        // generate message       
        GetBits(i, vecmu, p128.N);
        for (int j = 0; j < p128.N; j++)
        {
            vecmu[j] = mutoT(vecmu[j], p128.Msize);
        }

        // encryption
        trlweSymEnc(vecmu, trlwekey, p128.ks_stdev, p128, c);
        
        // attention !
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                database[i * 2 * p128.N + j * p128.N + k] = c[j][k];
            }
        }

        // trlweSymDec(c, trlwekey, p128, vecmu);
        // cout << GetNum(vecmu, p128.N) << endl;
    }

    free(vecmu);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
}


void SampleExtractIndex(uint32_t **trlwecipher, int index, Params128 &p128, uint32_t *tlwecipher)
{
    for (int i = 0; i <= index; i++)
    {
        tlwecipher[i + 1] = trlwecipher[1][index - i];
    }
    for (int i = index + 1; i < p128.N; i++)
    {
        tlwecipher[i + 1] = (uint32_t)(-trlwecipher[1][p128.N + index - i]);
    }     
    tlwecipher[0] = trlwecipher[0][index];
}

void Test_Sampleextract()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    uint32_t *mu = (uint32_t*)malloc(sizeof(uint32_t) * p128.N);
    for (int i = 0; i < p128.N; i++)
    {
        if(i % 2 == 0)
        {
            mu[i] = mutoT(1, p128.Msize);
        }
        else{
            mu[i] = mutoT(0, p128.Msize);
        }
    }
    // key generation
    uint32_t *trlwekey = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    trlweKeyGen(trlwekey, p128.N);

    uint32_t **c = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    trlweSymEnc(mu, trlwekey, p128.ks_stdev, p128, c);
    uint32_t *nc = (uint32_t*)malloc((p128.N + 1) * sizeof(uint32_t));

    SampleExtractIndex(c, 101, p128, nc);
    uint32_t msg;
    msg = tlweSymDec(nc, (int32_t*)trlwekey, p128.N, p128);
    cout << "\nSample Extraction result is : " << msg << endl;


    free(mu);
    free(trlwekey);
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
    free(nc);
}


void kskGen(int32_t *keyN, int32_t *keyn, Params128 &p128, uint32_t ****ksk)
{
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int l = 0; l < p128.n + 1; l++)
            {
                ksk[i][j][0][l] = 0;
            }
            for (int k = 1; k < (1 << p128.ks_basebit); k++)
            {
                tlweSymEnc(keyN[i] * k * (1 << (32 - (j + 1) * p128.ks_basebit)), keyn, p128.n, p128, ksk[i][j][k]);
            }    
        }    
    }    
}

void calc_rhs(uint32_t ****ksk, uint32_t *aibar, int mask, Params128 &p128, uint32_t *rhs)
{
    int *indices_rhs = (int*)malloc(p128.ks_t * sizeof(int));
    for (int i = 0; i < p128.ks_t; i++)
    {
        indices_rhs[i] = p128.ks_basebit * (p128.ks_t - 1 - i);
    }
    uint32_t *indices = (uint32_t*)malloc(p128.ks_t * p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            indices[i * p128.ks_t + j] = aibar[i] >> indices_rhs[j] & mask;
        }    
    }
        
    int *first = (int*)malloc(p128.N * p128.ks_t * sizeof(int));
    int *second = (int*)malloc(p128.N * p128.ks_t * sizeof(int));
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            first[i * p128.ks_t + j] = i;
            second[i * p128.ks_t + j] = j;
        }    
    }  

    // attention!
    for (int j = 0; j < p128.n + 1; j++)
    {
        rhs[j] = 0;
        for (int i = 0; i < p128.N * p128.ks_t; i++)
        {
            rhs[j] += ksk[first[i]][second[i]][indices[i]][j]; 
        }
    }
  
    free(indices_rhs);
    free(indices);
    free(first);
    free(second);
}

void IdentityKeySwitch(uint32_t *tlwecipher, uint32_t ****ksk, Params128 &p128, uint32_t *rhs)
{
    uint32_t *aibar = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        aibar[i] = uint32_t(round(tlwecipher[i + 1] * pow(2, p128.ks_basebit * p128.ks_t - 32)));
    }
    int mask = (1 << p128.ks_basebit) - 1;
    calc_rhs(ksk, aibar, mask, p128, rhs);
    rhs[0] = rhs[0] + tlwecipher[0];
    free(aibar);
}

void Test_IdentityKeySwitch()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    uint32_t mu = mutoT(1, p128.Msize);

    int32_t *key = (int32_t*)malloc(p128.N * sizeof(int32_t));
    int32_t *tkey = (int32_t*)malloc(p128.n * sizeof(int32_t));
    tlweKeyGen(key, p128.N);
    tlweKeyGen(tkey, p128.n);

    uint32_t ****ksk = (uint32_t****)malloc(p128.N * sizeof(uint32_t***));
    for (int i = 0; i < p128.N; i++)
    {
        ksk[i] = (uint32_t***)malloc(p128.ks_t * sizeof(uint32_t**));
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            ksk[i][j] = (uint32_t**)malloc((1 << p128.ks_basebit) * sizeof(uint32_t*));
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                ksk[i][j][k] = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
            }
        }    
    }
    kskGen(key, tkey, p128, ksk);

    uint32_t *tlwecipher = (uint32_t*)malloc((p128.N + 1) * sizeof(uint32_t));
    tlweSymEnc(mu, key, p128.N, p128, tlwecipher);

    uint32_t *nc = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    IdentityKeySwitch(tlwecipher, ksk, p128, nc);

    uint32_t msg = tlweSymDec(nc, tkey, p128.n, p128);

    cout << "\n Identity Key Switch msg is " << msg << endl;

    free(key);
    free(tkey);
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                free(ksk[i][j][k]);
            }
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            free(ksk[i][j]);
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        free(ksk[i]);
    }
    free(ksk);
    free(tlwecipher);
    free(nc);
}


void bkfft(int32_t *tlwekey, uint32_t *trlwekey, Params128 &p128, cuDoubleComplex ****bk)
{
    uint64_t *mu0 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    uint64_t *mu1 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu0[i] = 0;
        mu1[i] = 0;
    }
    mu1[0] = _two32;
    for (int i = 0; i < p128.n; i++)
    {
        if(tlwekey[i] == 0)
        {
            trgswfftSymEnc(mu0, trlwekey, p128, bk[i]);
        }
        else{
            trgswfftSymEnc(mu1, trlwekey, p128, bk[i]);
        }
    }
    free(mu0);
    free(mu1);
}



void bkfft64(int32_t *tlwekey, uint64_t *trlwekey, Params128 &p128, cuDoubleComplex ****bk)
{
    for (int i = 0; i < p128.n; i++)
    {
        if(tlwekey[i] == 0){
            trgswfftSymEnc64(0, trlwekey, p128, bk[i]);
        }
        else{
            trgswfftSymEnc64(_two64, trlwekey, p128, bk[i]);
        }
    }  
}




void PolynomialMulByXai(uint32_t *poly, int a, int N, int64_t *newpoly)
{
    a = a % (N + N);
    if(a == 0)
    {
        for (int i = 0; i < N; i++)
        {
            newpoly[i] = poly[i];
        }
    }
    else if (a < N)
    {
        for (int i = 0; i < a; i++)
        {
            newpoly[i] = (_two32 - poly[N - a + i]) % _two32;
        }
        for (int i = a; i < N; i++)
        {
            newpoly[i] = poly[i - a] % _two32;
        }
    }
    else{
        for (int i = 0; i < a - N; i++)
        {
            newpoly[i] = poly[N + N- a + i] % _two32;
        }
        for (int i = a - N; i < N; i++)
        {
            newpoly[i] = (_two32 - poly[i - a + N]) % _two32;
        }
    }
}

void PolynomialMulByXai(int64_t *poly, int a, int N, int64_t *newpoly)
{
    a = a % (N + N);
    if(a == 0)
    {
        for (int i = 0; i < N; i++)
        {
            newpoly[i] = poly[i];
        }
    }
    else if (a < N)
    {
        for (int i = 0; i < a; i++)
        {
            newpoly[i] = (_two32 - poly[N - a + i]) % _two32;
        }
        for (int i = a; i < N; i++)
        {
            newpoly[i] = poly[i - a] % _two32;
        }
    }
    else{
        for (int i = 0; i < a - N; i++)
        {
            newpoly[i] = poly[N + N- a + i] % _two32;
        }
        for (int i = a - N; i < N; i++)
        {
            newpoly[i] = (_two32 - poly[i - a + N]) % _two32;
        }
    }
}

void PolynomialMulByXai64(uint64_t *poly, int a, int N, uint64_t *newpoly)
{
    a = a % (N + N);
    cout << "\na:" << a << endl;
    cout << "\npoly\n";
    for (int i = 0; i < N; i++)
    {
        cout << poly[i] << ", ";
    }
    cout << endl;
    if(a == 0 || a == N + N)
    {
        for (int i = 0; i < N; i++)
        {
            newpoly[i] = poly[i];
        }
    }
    if(a < 0)
    {
        a = N + N + a;
    }
    if(a > N + N)
    {
        a = a - N - N;
    }
    if (a < N)
    {
        for (int i = 0; i < a; i++)
        {
            newpoly[i] = -poly[N - a + i];
        }
        for (int i = a; i < N; i++)
        {
            newpoly[i] = poly[i - a];
        }
    }
    else{
        for (int i = 0; i < a - N; i++)
        {
            newpoly[i] = poly[N + N- a + i];
        }
        for (int i = a - N; i < N; i++)
        {
            newpoly[i] = -poly[i - a + N];
        }
    }
    cout << "\nrotate\n";
    for (int i = 0; i < N; i++)
    {
        cout << newpoly[i] << ", ";
    }
    cout << endl;
}

void PolynomialMulByXai64(uint64_t *poly, int a, int N, double *newpoly)
{
    a = a % (N + N);
    cout << "\na:" << a << endl;
     cout << "\npoly\n";
    for (int i = 0; i < N; i++)
    {
        cout << poly[i] << ", ";
    }
    cout << endl;
    if(a == 0 || a == N + N)
    {
        for (int i = 0; i < N; i++)
        {
            newpoly[i] = poly[i];
        }
    }
    if(a < 0)
    {
        a = N + N + a;
    }
    if(a > N + N)
    {
        a = a - N - N;
    }
    if (a < N)
    {
        for (int i = 0; i < a; i++)
        {
            newpoly[i] = (-1) * (double)poly[N - a + i];
        }
        for (int i = a; i < N; i++)
        {
            newpoly[i] = poly[i - a];
        }
    }
    else{
        for (int i = 0; i < a - N; i++)
        {
            newpoly[i] = poly[N + N- a + i];
        }
        for (int i = a - N; i < N; i++)
        {
            newpoly[i] = (-1) * (double)poly[i - a + N];
        }
    }
    cout << "\nrotate\n";
    for (int i = 0; i < N; i++)
    {
        cout << newpoly[i] << ", ";
    }
    cout << endl;
}

// in use
void PolynomialMulByXai64(double *poly, int a, int N, double *newpoly)
{
    a = a % (N + N);
    cout << "\na:" << a << endl;
     cout << "\npoly\n";
    for (int i = 0; i < N; i++)
    {
        cout << poly[i] << ", ";
    }
    cout << endl;
    if(a == 0 || a == N + N)
    {
        for (int i = 0; i < N; i++)
        {
            newpoly[i] = poly[i];
        }
    }
    if(a < 0)
    {
        a = N + N + a;
    }
    if(a > N + N)
    {
        a = a - N - N;
    }
    if (a < N)
    {
        for (int i = 0; i < a; i++)
        {
            newpoly[i] = (-1) * (double)poly[N - a + i];
        }
        for (int i = a; i < N; i++)
        {
            newpoly[i] = poly[i - a];
        }
    }
    else{
        for (int i = 0; i < a - N; i++)
        {
            newpoly[i] = poly[N + N- a + i];
        }
        for (int i = a - N; i < N; i++)
        {
            newpoly[i] = (-1) * (double)poly[i - a + N];
        }
    }
    cout << "\nrotate newpoly\n";
    for (int i = 0; i < N; i++)
    {
        cout << newpoly[i] << ", ";
    }
    cout << endl;
}

void BlindRotateFFT(cuDoubleComplex ****bkfft, uint32_t *tlwe, uint32_t **trlwe, Params128 &p128, int64_t **acc)
{
    int TN = 2 * p128.N;
    uint32_t *bara = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    for (int i = 0; i < p128.n + 1; i++)
    {
        bara[i] = uint32_t(round(double(tlwe[i]) * pow(2, -32) * 2 * p128.N)) % (2 * p128.N);
    }
    
    PolynomialMulByXai(trlwe[0], TN - bara[0], p128.N, acc[0]);
    PolynomialMulByXai(trlwe[1], TN - bara[0], p128.N, acc[1]);

    int64_t **tmp = (int64_t**)malloc(2 * sizeof(int64_t*));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (int64_t*)malloc(p128.N * sizeof(int64_t));
    }
    int64_t **cprod = (int64_t**)malloc(2 * sizeof(int64_t*));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (int64_t*)malloc(p128.N * sizeof(int64_t));
    }
    for (int i = 0; i < p128.n; i++)
    {
        
        if (bara[i + 1] == 0)
            continue;
        PolynomialMulByXai(acc[0], TN - bara[i + 1], p128.N, tmp[0]);
        PolynomialMulByXai(acc[1], TN - bara[i + 1], p128.N, tmp[1]);

        CMUXFFT(bkfft[i], tmp, acc, cprod, p128);

        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.N; k++)
            {
                acc[j][k] = cprod[j][k];
            }
        }    
    }    

    free(bara);
    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
}

void BlindRotateFFT64(cuDoubleComplex ****bkfft, uint32_t *tlwe, uint64_t **trlwe, Params128 &p128, uint64_t **acc)
{
    int32_t *bara = (int32_t*)malloc((p128.n + 1) * sizeof(int32_t));
    cout << "\nbara\n";
    for (int i = 0; i < p128.n + 1; i++)
    {
        bara[i] = int32_t(round(double(tlwe[i]) * pow(2, -32) * 2 * p128.nbar)) % (2 * p128.nbar);
        cout << bara[i] << ", ";
    }
    cout << endl;
    
    double **acc0 = (double**)malloc(2 * sizeof(double*));
    for (int i = 0; i < 2; i++)
    {
        acc0[i] = (double*)malloc(p128.nbar * sizeof(double));
    }  
    double **acc1 = (double**)malloc(2 * sizeof(double*));
    for (int i = 0; i < 2; i++)
    {
        acc1[i] = (double*)malloc(p128.nbar * sizeof(double));
    }
    uint64_t **tmp = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }
    uint64_t **cprod = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        cprod[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }

    PolynomialMulByXai64(trlwe[0], bara[0], p128.nbar, acc0[0]);
    PolynomialMulByXai64(trlwe[1], bara[0], p128.nbar, acc0[1]);
    cout << "\nacc0\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cout << acc0[i][j] << ", ";
        }
        cout << endl;
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            acc[i][j] = DoubleToUint64(acc0[i][j]);
        }    
    }
    
    
    
    bool flag = false;
    for (int i = 0; i < p128.n; i++)
    {
        
        if (bara[i + 1] == 0)
            continue;
        if(!flag)
        {
            PolynomialMulByXai64(acc0[0], bara[i + 1], p128.nbar, acc1[0]);
            PolynomialMulByXai64(acc0[1], bara[i + 1], p128.nbar, acc1[1]);
            cout << "\ntmp" << i + 1 << endl;
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.nbar; k++)
                {
                    cout << acc1[j][k] << ", ";
                }
                cout << endl;
            }
            cout << endl;
            
            CMUXFFT64(bkfft[i], acc1, acc0, cprod, p128);
            flag = true;
            // cout << "\noutside cprod\n";
            // for (int j = 0; j < 2; j++)
            // {
            //     for (int k = 0; k < p128.nbar; k++)
            //     {
            //         cout << cprod[i][j] << ", ";
            //     }
            //     cout << endl;
            // }    
            // cout << endl;
        }
        else{
            PolynomialMulByXai64(acc[0], bara[i + 1], p128.nbar, tmp[0]);
            PolynomialMulByXai64(acc[1], bara[i + 1], p128.nbar, tmp[1]);
            cout << "\ntmp" << i + 1 << endl;
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.nbar; k++)
                {
                    cout << tmp[j][k] << ", ";
                }
                cout << endl;
            }
            cout << endl;
            CMUXFFT64(bkfft[i], tmp, acc, cprod, p128);
        }
        cout << "\nacc" << i +  1 << "\n";
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < p128.nbar; k++)
            {
                acc[j][k] = cprod[j][k];
                cout << cprod[j][k] << ", ";
            }
            cout << endl;
        }    
        cout << endl;
    }    

    free(bara);
    for (int i = 0; i < 2; i++)
    {
        free(acc0[i]);
    }
    free(acc0);
    for (int i = 0; i < 2; i++)
    {
        free(acc1[i]);
    }
    free(acc1);
    for (int i = 0; i < 2; i++)
    {
        free(tmp[i]);
    }
    free(tmp);
    for (int i = 0; i < 2; i++)
    {
        free(cprod[i]);
    }
    free(cprod);
}


void Test_BlindRotateFFT()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
    uint32_t mu = mutoT(1, p128.Msize);

    int32_t *tlwekey = (int32_t*)malloc(p128.n * sizeof(int32_t));
    uint32_t *trlwekey = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen(trlwekey, p128.N);

    // bk size: p128.n * (2 * p128.bk_l) * 2 * (p128.N / 2)     (630,4,2,512)
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ****bk = (cuDoubleComplex****)malloc(p128.n * sizeof(cuDoubleComplex***));
    for (int i = 0; i < p128.n; i++)
    {
        bk[i] = (cuDoubleComplex***)malloc(lines * sizeof(cuDoubleComplex**));
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            bk[i][j] = (cuDoubleComplex**)malloc(2 * sizeof(cuDoubleComplex*));
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                bk[i][j][k] = (cuDoubleComplex*)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
            }
        }
    }
    bkfft(tlwekey, trlwekey, p128, bk);

    uint32_t *tlwecipher = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(mu, tlwekey, p128.n, p128, tlwecipher);
    
    uint32_t **trlwecipher = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    
    for (int i = 0; i < p128.N / 2; i++)
    {
        trlwecipher[0][i] = mutoT(-1, 4);
    }
    for (int i = p128.N / 2; i < p128.N; i++)
    {
        trlwecipher[0][i] = mutoT(1, 4);
    }
    for (int i = 0; i < p128.N; i++)
    {
        trlwecipher[1][i] = 0;
    }

    int64_t **acc = (int64_t**)malloc(2 * sizeof(int64_t*));
    for (int i = 0; i < 2; i++)
    {
        acc[i] = (int64_t*)malloc(p128.N * sizeof(int64_t));
    }
    BlindRotateFFT(bk, tlwecipher, trlwecipher, p128, acc);

    uint32_t *msg1 = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    
    uint32_t **ans = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        ans[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            ans[i][j] = acc[i][j];
        }    
    }
    trlweSymDec(ans, trlwekey, p128, msg1);
    cout << "\nBlindRotateFFT msg is " << "\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg1[i] << " ";
    }
    cout << endl;
     

    free(tlwekey);
    free(trlwekey);
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                free(bk[i][j][k]);
            }
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            free(bk[i][j]);
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        free(bk[i]);
    }
    free(bk);
    free(tlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(acc[i]);
    }
    free(acc);
    for (int i = 0; i < 2; i++)
    {
        free(ans[i]);
    }
    free(ans);
    free(msg1);   
}

void Test_BlindRotateFFT64()
{
    Params128 p128 = Params128(2, 2, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 4);
    uint32_t mu = mutoT(1, 2);
    cout << "\nmu\n" << mu << endl;
    int32_t *tlwekey = (int32_t*)malloc(p128.n * sizeof(int32_t));
    uint64_t *trlwekey = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen64(trlwekey, p128.nbar);
    cout << "\ntlwekey\n";
    for (int i = 0; i < p128.n; i++)
    {
        cout << tlwekey[i] << ", ";
    }
    cout << endl;
    cout << "\ntrlwekey\n";
    for (int i = 0; i < p128.nbar; i++)
    {
        cout << trlwekey[i] << ", ";
    }
    cout << endl;

    // bk size: (p128.n, 2 * p128.bk_lbar, 2, p128.nbar / 2)     (630,8,2,1024)
    int lines = 2 * p128.bk_lbar;
    cuDoubleComplex ****bk = (cuDoubleComplex****)malloc(p128.n * sizeof(cuDoubleComplex***));
    for (int i = 0; i < p128.n; i++)
    {
        bk[i] = (cuDoubleComplex***)malloc(lines * sizeof(cuDoubleComplex**));
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            bk[i][j] = (cuDoubleComplex**)malloc(2 * sizeof(cuDoubleComplex*));
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                bk[i][j][k] = (cuDoubleComplex*)malloc(p128.nbar / 2 * sizeof(cuDoubleComplex));
            }
        }
    }
    bkfft64(tlwekey, trlwekey, p128, bk);
    cout << "\n bk\n";
    for (int i = 0; i < p128.n; i++)
    {
        cout << "[";
        for (int j = 0; j < lines; j++)
        {
            cout << "[";
            for (int k = 0; k < 2; k++)
            {
                cout << "[";
                for (int l = 0; l < p128.nbar / 2; l++)
                {
                    cout << bk[i][j][k][l].x << "+" << bk[i][j][k][l].y << "j";
                }
                cout << "]";
                if(k == 0){
                    cout << ", ";
                }
            }
            cout << "],\n";
        }
        cout << "],\n";
    }
    cout << endl;


    uint32_t *tlwecipher = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(mu, tlwekey, p128.n, p128, tlwecipher);
    cout << "\ntlwecipher\n";
    for (int i = 0; i < p128.n + 1; i++)
    {
        cout << tlwecipher[i] << ", ";
    }
    cout << endl;

    
    uint64_t **trlwecipher = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }
    
    for (int i = 0; i < p128.nbar / 2; i++)
    {
        trlwecipher[0][i] = dtot64((double)-1/(double)4);
        trlwecipher[1][i] = 0;
    }
    for (int i = p128.nbar / 2; i < p128.nbar; i++)
    {
        trlwecipher[0][i] = dtot64((double)1/(double)4);
        trlwecipher[1][i] = 0;
    }
    cout << "\ntrlwecipher\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            cout << trlwecipher[i][j] << ", ";
        }    
        cout << endl;
    }
    cout << endl;
    


    uint64_t **acc = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        acc[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }
    BlindRotateFFT64(bk, tlwecipher, trlwecipher, p128, acc);

    uint64_t *msg1 = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    
    uint64_t **ans = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        ans[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }
    cout << "\nfinal acc\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar; j++)
        {
            ans[i][j] = acc[i][j];
            cout << ans[i][j] << ", ";
        }    
        cout << endl;
    }
    cout << endl;
    trlweSymDec64(ans, trlwekey, p128, msg1);
    cout << "\nBlindRotateFFT msg is " << "\n";
    for (int i = 0; i < p128.nbar; i++)
    {
        cout << msg1[i] << " ";
    }
    cout << endl;
     

    free(tlwekey);
    free(trlwekey);
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                free(bk[i][j][k]);
            }
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            free(bk[i][j]);
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        free(bk[i]);
    }
    free(bk);
    free(tlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
    for (int i = 0; i < 2; i++)
    {
        free(acc[i]);
    }
    free(acc);
    for (int i = 0; i < 2; i++)
    {
        free(ans[i]);
    }
    free(ans);
    free(msg1);   
}



void privkskGen(int32_t *keyn, uint32_t *keyN, Params128 &p128, uint32_t ******privksk)
{
    uint32_t *veczero = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        veczero[i] = 0;
    }
    uint32_t **trlwecipher = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    
    for (int z = 0; z < 2; z++)
    {
        for (int i = 0; i < p128.n + 1; i++)
        {
            for (int j = 0; j < p128.ks_t; j++)
            {
                // u = 0      
                for (int n = 0; n < p128.N; n++)
                {
                    privksk[z][i][j][0][0][n] = 0;
                    privksk[z][i][j][0][1][n] = 0;
                }     

                for (int u = 1; u < (1 << p128.ks_basebit); u++)
                {
                    trlweSymEnc(veczero, keyN, p128.ks_stdev, p128, trlwecipher);
                    for (int n = 0; n < p128.N; n++)
                    {
                        privksk[z][i][j][u][0][n] = trlwecipher[0][n];
                        privksk[z][i][j][u][1][n] = trlwecipher[1][n];                    
                    }  
                    privksk[z][i][j][u][0][z] += u * keyn[i] << (32 - (j + 1) * p128.ks_basebit);
                }    
            }    
        }    
    }
    
    free(veczero);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
}


// trlwekey(nbar), trlwekey(N)   (2, 2049, 10, 8, 2, 1024)
void privkskGen2(uint64_t *keyn, uint32_t *keyN, Params128 &p128, uint32_t ******privksk)
{
    uint32_t *key = (uint32_t*)malloc((p128.nbar + 1) * sizeof(uint32_t));
    for (int i = 0; i < p128.nbar; i++)
    {
        key[i + 1] = (uint32_t)keyn[i];
    }
    key[0] = 1;

    uint32_t *veczero = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        veczero[i] = 0;
    }
    uint32_t **trlwecipher = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        trlwecipher[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    
    for (int z = 0; z < 2; z++)
    {
        for (int i = 0; i < p128.n + 1; i++)
        {
            for (int j = 0; j < p128.ks_tbar; j++)
            {
                // u = 0      
                for (int n = 0; n < p128.N; n++)
                {
                    privksk[z][i][j][0][0][n] = 0;
                    privksk[z][i][j][0][1][n] = 0;
                }     

                for (int u = 1; u < (1 << p128.ks_basebitbar); u++)
                {
                    trlweSymEnc(veczero, keyN, p128.ks_stdevbar, p128, trlwecipher);
                    for (int n = 0; n < p128.N; n++)
                    {
                        privksk[z][i][j][u][0][n] = trlwecipher[0][n];
                        privksk[z][i][j][u][1][n] = trlwecipher[1][n];                    
                    }  
                    privksk[z][i][j][u][z][0] += u * key[i] << (32 - (j + 1) * p128.ks_basebitbar);
                }    
            }    
        }    
    }
    
    free(key);
    free(veczero);
    for (int i = 0; i < 2; i++)
    {
        free(trlwecipher[i]);
    }
    free(trlwecipher);
}


void TLWE2TRLWEprivateKeySwitch(uint32_t *tlwecipher, uint32_t ******privksk, int u, Params128 &p128, uint32_t **cc)
{
    uint32_t *aibar = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    for (int i = 0; i < p128.n + 1; i++)
    {
        aibar[i] = uint32_t(round(tlwecipher[i] * pow(2, p128.ks_basebit * p128.ks_t -32)));
    }
    int mask = (1 << p128.ks_basebit) - 1;
    uint32_t *indices_rhs = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.ks_t; i++)
    {
        indices_rhs[i] = p128.ks_basebit * (p128.ks_t - i - 1);
    }
    uint32_t **indices_lhs = (uint32_t**)malloc((p128.n + 1) * sizeof(uint32_t*));
    for (int i = 0; i < p128.n + 1; i++)
    {
        indices_lhs[i] = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.n + 1; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            indices_lhs[i][j] = aibar[i];
        }    
    }
    uint32_t **indices = (uint32_t**)malloc((p128.n + 1) * sizeof(uint32_t*));
    for (int i = 0; i < p128.n + 1; i++)
    {
        indices[i] = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.n + 1; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            indices[i][j] = indices_lhs[i][j] >> indices_rhs[j] & mask;
        }    
    }
    uint32_t *second = (uint32_t*)malloc((p128.n + 1) * p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.n + 1; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            second[i * p128.ks_t + j] = i;
        }    
    }
    uint32_t *third = (uint32_t*)malloc((p128.n + 1) * p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.n + 1; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            third[i * p128.ks_t + j] = j;
        }    
    }

    int i, j, k;
    for (j = 0; j < 2; j++)
    {
        for (k = 0; k < p128.N; k++)
        {
            cc[j][k] = 0;
            for (i = 0; i < (p128.n + 1) * p128.ks_t; i++)
            {
                cc[j][k] += privksk[u][second[i]][third[i]][indices[i / p128.ks_t][i % p128.ks_t]][j][k];
            }
        }    
    } 
    
    

    free(aibar);
    free(indices_rhs);
    for (int i = 0; i < p128.n + 1; i++)
    {
        free(indices_lhs[i]);
    }
    free(indices_lhs);
    for (int i = 0; i < p128.n + 1; i++)
    {
        free(indices[i]);
    }
    free(indices);
    free(second);
    free(third);
}

void TLWE2TRLWEprivateKeySwitch_CB()
{

}


void Test_TLWE2TRLWE()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    uint32_t *mu = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.N; i++)
    {
        if (i % 2 == 0)
        {
            mu[i] = mutoT(1, p128.Msize);
        }
        else{
            mu[i] = mutoT(0, p128.Msize);
        }
    }

    uint32_t *key = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    int32_t *tkey = (int32_t *)malloc((p128.n + 1) * sizeof(int32_t));
    int32_t *enckey = (int32_t *)malloc(p128.n * sizeof(uint32_t));
    trlweKeyGen(key, p128.N);
    tlweKeyGen(tkey, p128.n + 1);
    tkey[0] = 1;
    for (int i = 1; i < p128.n + 1; i++)
    {
        enckey[i - 1] = tkey[i];
    }



    uint32_t *tlwecipher = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(mu[0], enckey, p128.n, p128, tlwecipher);
  
    
    // privksk size :(2, n + 1, p128.ks_t, 1 << p128.ks_basebit , 2, N)
    uint32_t ******privksk = (uint32_t******)malloc(2 * sizeof(uint32_t*****));
    for (int i = 0; i < 2; i++)
    {
        privksk[i] = (uint32_t*****)malloc((p128.n + 1) * sizeof(uint32_t****));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            privksk[i][j] = (uint32_t****)malloc(p128.ks_t * sizeof(uint32_t***));
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                privksk[i][j][k] = (uint32_t***)malloc((1 << p128.ks_basebit) * sizeof(uint32_t**));
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebit); l++)
                {
                    privksk[i][j][k][l] = (uint32_t**)malloc(2 * sizeof(uint32_t*));
                }    
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebit); l++)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        privksk[i][j][k][l][m] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
                    }    
                }    
            }    
        }    
    }
    
    privkskGen(tkey, key, p128, privksk);
    
    uint32_t **cc1 = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        cc1[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    uint32_t **cc0 = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        cc0[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    int u = 1;
    TLWE2TRLWEprivateKeySwitch(tlwecipher, privksk, u, p128, cc1);
    u = 0;
    TLWE2TRLWEprivateKeySwitch(tlwecipher, privksk, u, p128, cc0);
    

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            cc0[i][j] = (uint32_t)(cc0[i][j] + cc1[i][j]);
        }    
    }
    uint32_t *msg = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(cc0, key, p128, msg);
    cout << "\nTLWE2TRLWE decryption result\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    cout << endl;
    

    free(mu);
    free(key);
    free(tkey);
    free(enckey);
    free(tlwecipher);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebit); l++)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        free(privksk[i][j][k][l][m]);
                    }    
                }    
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebit); l++)
                {
                    free(privksk[i][j][k][l]);
                }    
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                free(privksk[i][j][k]);
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.n + 1; j++)
        {
            free(privksk[i][j]);
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        free(privksk[i]);
    }
    free(privksk);
    for (int i = 0; i < 2; i++)
    {
        free(cc1[i]);
    }
    free(cc1);
    for (int i = 0; i < 2; i++)
    {
        free(cc0[i]);
    }
    free(cc0);
    free(msg);
}

void kskGenTR(int32_t *keyn, uint32_t *keyN, Params128 &p128, uint32_t *****ksk)
{
    uint32_t *vecmu = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            // k = 0
            for (int n = 0; n < p128.N; n++)
            {
                ksk[i][j][0][0][n] = 0;
                ksk[i][j][0][1][n] = 0;
            }

            for (int k = 1; k < (1 << p128.ks_basebit); k++)
            {
                vecmu[0] = keyn[i] * k * (1 << (32 - (j + 1) * p128.ks_basebit));
                for (int l = 1; l < p128.N; l++)
                {
                    vecmu[l] = 0;
                }
                trlweSymEnc(vecmu, keyN, p128.ks_stdev, p128, ksk[i][j][k]);    
            }    
        }    
    }   
    free(vecmu);
}


void PubKS(uint32_t **tlcipherlist, int listlen, uint32_t *****ksk, Params128 &p128, uint32_t **acc)
{
    uint32_t *indices_rhs = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.ks_t; i++)
    {
        indices_rhs[i] = p128.ks_basebit * (p128.ks_t - i - 1);
    }
    int mask = (1 << p128.ks_basebit) - 1;
    uint32_t *first = (uint32_t*)malloc(p128.n * p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            first[i * p128.ks_t + j] = i;
        }    
    }
    uint32_t *second = (uint32_t*)malloc(p128.n * p128.ks_t * sizeof(uint32_t));
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            second[i * p128.ks_t + j] = j;
        }    
    }

    uint32_t *aibar = (uint32_t*)malloc(p128.n * sizeof(uint32_t));
    uint32_t **indices_lhs = (uint32_t**)malloc(p128.n * sizeof(uint32_t*));
    for (int i = 0; i < p128.n; i++)
    {
        indices_lhs[i] = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    }
    uint32_t **indices = (uint32_t**)malloc(p128.n * sizeof(uint32_t*));
    for (int i = 0; i < p128.n; i++)
    {
        indices[i] = (uint32_t*)malloc(p128.ks_t * sizeof(uint32_t));
    }
    uint32_t **t = (uint32_t**)malloc(p128.n * p128.ks_t * sizeof(uint32_t*));
    for (int i = 0; i < p128.n * p128.ks_t; i++)
    {
        t[i] = (uint32_t*)malloc(2 * p128.N * sizeof(uint32_t));
    }  
    uint32_t *rhs = (uint32_t*)malloc(2 * p128.N * sizeof(uint32_t));
    uint32_t *lhs = (uint32_t*)malloc(2 * p128.N * sizeof(uint32_t));
    for (int i = 0; i < 2 * p128.N; i++)
    {
        lhs[i] = 0;
    }
    uint32_t **ac = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        ac[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    int64_t **acxp = (int64_t**)malloc(2 * sizeof(int64_t*));
    for (int i = 0; i < 2; i++)
    {
        acxp[i] = (int64_t*)malloc(p128.N * sizeof(int64_t));
    }
    
    

    float Conaibar = pow(2.0, p128.ks_basebit * p128.ks_t - 32);
    int TN = 2 * p128.N;


    for (int i = 0; i < listlen; i++)
    {
        for (int j = 0; j < p128.n; j++)
        {
            aibar[j] = uint32_t(round(tlcipherlist[i][j + 1] * Conaibar));
        }
        for (int j = 0; j < p128.n; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                indices_lhs[j][k] = aibar[j];
            }    
        }
        for (int j = 0; j < p128.n; j++)
        {
            for (int k = 0; k < p128.ks_t; k++)
            {
                indices[j][k] = indices_lhs[j][k] >> indices_rhs[k] & mask;
            }    
        }

        int a, b, c;
        for (a = 0; a < p128.n * p128.ks_t; a++)
        {        
            for (b = 0; b < 2; b++)
            {
                for (c = 0; c < p128.N; c++)
                {
                    t[a][b * p128.N + c] = ksk[first[a]][second[a]][indices[a / p128.ks_t][a % p128.ks_t]][b][c];
                }    
            }    
        }    

        for (a = 0; a < 2 * p128.N; a++)
        {
            rhs[a] = 0;
            for (b = 0; b < p128.n * p128.ks_t; b++)
            {
                rhs[a] += t[b][a];
            }    
        }
        lhs[0] = tlcipherlist[i][0];
        for (int j = 0; j < 2 * p128.N; j++)
        {
            ac[j / p128.N][j % p128.N] = lhs[j] + rhs[j];
        }


        if(i == 0){
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.N; k++)
                {
                    acc[j][k] = ac[j][k];
                }    
            }
        }
        else{
            PolynomialMulByXai(ac[0], TN + i, p128.N, acxp[0]);
            PolynomialMulByXai(ac[1], TN + i, p128.N, acxp[1]);
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < p128.N; k++)
                {
                    acc[j][k] = acc[j][k] + acxp[j][k];
                }    
            }    
        }    
    }
    
    

    free(indices_rhs);
    free(first);
    free(second);
    free(aibar);
    for (int i = 0; i < p128.n; i++)
    {
        free(indices_lhs[i]);
    }
    free(indices_lhs);
    for (int i = 0; i < p128.n; i++)
    {
        free(indices[i]);
    }
    free(indices);
    for (int i = 0; i < p128.n * p128.ks_t; i++)
    {
        free(t[i]);
    }
    free(t);
    free(rhs);
    free(lhs);
    for (int i = 0; i < 2; i++)
    {
        free(ac[i]);
    }
    free(ac);
    for (int i = 0; i < 2; i++)
    {
        free(acxp[i]);
    }
    free(acxp);
}



void Test_PubKS()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
    uint32_t mu0 = mutoT(3, p128.Msize);
    uint32_t mu1 = mutoT(5, p128.Msize);
   
    int32_t *tlwekey = (int32_t *)malloc(p128.n * sizeof(int32_t));
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen(trlwekey, p128.N);

    // ksk size: (p128.n, p128.ks_t, 1 << p128.ks_basebit, 2, p128.N)
    uint32_t *****ksk = (uint32_t*****)malloc(p128.n * sizeof(uint32_t****));
    for (int i = 0; i < p128.n; i++)
    {
        ksk[i] = (uint32_t****)malloc(p128.ks_t * sizeof(uint32_t***));
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            ksk[i][j] = (uint32_t***)malloc((1 << p128.ks_basebit) * sizeof(uint32_t**));
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                ksk[i][j][k] = (uint32_t**)malloc(2 * sizeof(uint32_t*));
            }    
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                for (int l = 0; l < 2; l++)
                {
                    ksk[i][j][k][l] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
                }    
            }    
        }
    }
    kskGenTR(tlwekey, trlwekey, p128, ksk);

    uint32_t **tlcipherlist = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        tlcipherlist[i] = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    }
    
    tlweSymEnc(mu0, tlwekey, p128.n, p128, tlcipherlist[0]);
    tlweSymEnc(mu1, tlwekey, p128.n, p128, tlcipherlist[1]);

    uint32_t **acc = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        acc[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    PubKS(tlcipherlist, 2, ksk, p128, acc);

    uint32_t *msg = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    trlweSymDec(acc, trlwekey, p128, msg);
    cout << "\nTest PubKS decryption result\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    

    

  
    free(tlwekey);
    free(trlwekey);
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                for (int l = 0; l < 2; l++)
                {
                    free(ksk[i][j][k][l]);
                }    
            }    
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                free(ksk[i][j][k]);
            }    
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            free(ksk[i][j]);
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        free(ksk[i]);
    }
    free(ksk);
    for (int i = 0; i < 2; i++)
    {
        free(tlcipherlist[i]);
    }
    free(tlcipherlist);
    for (int i = 0; i < 2; i++)
    {
        free(acc[i]);
    }
    free(acc);
    free(msg);
}



void GateBootstrappingTLWE2TLWEFFT(uint32_t *tlwecipher, cuDoubleComplex ****bkfft, Params128 &p128, uint32_t *newcipher)
{
    uint32_t **testvec = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        testvec[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < p128.N / 2; i++)
    {
        testvec[0][i] = dtot32(-pow(2, -3));
        testvec[1][i] = 0;
    }
    for (int i = p128.N / 2; i < p128.N; i++)
    {
        testvec[0][i] = dtot32(pow(2, -3));
        testvec[1][i] = 0;
    }

    int64_t **acc = (int64_t**)malloc(2 * sizeof(int64_t*));
    for (int i = 0; i < 2; i++)
    {
        acc[i] = (int64_t*)malloc(p128.N * sizeof(int64_t));
    }
    BlindRotateFFT(bkfft, tlwecipher, testvec, p128, acc);

    uint32_t **nacc = (uint32_t**)malloc(2 * sizeof(uint32_t*));
    for (int i = 0; i < 2; i++)
    {
        nacc[i] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            nacc[i][j] = acc[i][j];
        }    
    }
    SampleExtractIndex(nacc, 0, p128, newcipher);
    

    for (int i = 0; i < 2; i++)
    {
        free(testvec[i]);
    }
    free(testvec);
    for (int i = 0; i < 2; i++)
    {
        free(acc[i]);
    }
    free(acc);
    for (int i = 0; i < 2; i++)
    {
        free(nacc[i]);
    }
    free(nacc);   
}

void GateBootstrappingTLWE2TLWEFFT_CB(uint32_t *tlwecipher, cuDoubleComplex ****bk, double mu, Params128 &p128, int64_t *tlwekey)
{
    uint64_t **testvec = (uint64_t**)malloc(2 * sizeof(uint64_t*));
    for (int i = 0; i < 2; i++)
    {
        testvec[i] = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    }
    for (int i = 0; i < p128.nbar / 2; i++)
    {
        testvec[0][i] = dtot64(-mu / 2);
        testvec[1][i] = 0;
    }
    for (int i = p128.nbar / 2; i < p128.nbar; i++)
    {
        testvec[0][i] = dtot64(mu / 2);
        testvec[1][i] = 0;
    }

    

    for (int i = 0; i < 2; i++)
    {
        free(testvec[i]);
    }
    free(testvec);
}

void Test_GateBootstrappingTLWE2TRLWEFFT()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
   
    int32_t *tlwekey = (int32_t *)malloc(p128.n * sizeof(int32_t));
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen(trlwekey, p128.N);

    // bk size: p128.n * (2 * p128.bk_l) * 2 * (p128.N / 2)     (630,4,2,512)
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ****bk = (cuDoubleComplex****)malloc(p128.n * sizeof(cuDoubleComplex***));
    for (int i = 0; i < p128.n; i++)
    {
        bk[i] = (cuDoubleComplex***)malloc(lines * sizeof(cuDoubleComplex**));
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            bk[i][j] = (cuDoubleComplex**)malloc(2 * sizeof(cuDoubleComplex*));
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                bk[i][j][k] = (cuDoubleComplex*)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
            }
        }
    }
    bkfft(tlwekey, trlwekey, p128, bk);

    uint32_t *tlwecipher = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(mutoT(p128.Msize / 2, p128.Msize), tlwekey, p128.n, p128, tlwecipher);

    uint32_t *newcipher = (uint32_t*)malloc((p128.N + 1) * sizeof(uint32_t));
    GateBootstrappingTLWE2TLWEFFT(tlwecipher, bk, p128, newcipher);
    newcipher[0] = uint32_t(newcipher[0] + dtot32(1.0 / 8.0));
    
    uint32_t msg = tlweSymDec(newcipher, (int32_t*)trlwekey, p128.N, p128);
    cout << "\nTest_GateBootstrappingTLWE2TLWEFFT decryption result is " << msg << endl;


    free(tlwekey);
    free(trlwekey);
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                free(bk[i][j][k]);
            }
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            free(bk[i][j]);
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        free(bk[i]);
    }
    free(bk);
    free(tlwecipher);
    free(newcipher);
}

void GateBootstrappingFFT(uint32_t *tlwecipher, cuDoubleComplex ****bk, uint32_t ****ksk, Params128 &p128, uint32_t *nc)
{
    uint32_t *newcipher = (uint32_t*)malloc((p128.N + 1) * sizeof(uint32_t));
    GateBootstrappingTLWE2TLWEFFT(tlwecipher, bk, p128, newcipher);
    newcipher[0] = uint32_t(newcipher[0] + dtot32(1.0 / 8.0));
    IdentityKeySwitch(newcipher, ksk, p128, nc);
    free(newcipher);
}

void Test_GateBootstrappingFFT()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
   
    int32_t *tlwekey = (int32_t *)malloc(p128.n * sizeof(int32_t));
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen(trlwekey, p128.N);

    // bk size: p128.n * (2 * p128.bk_l) * 2 * (p128.N / 2)     (630,4,2,512)
    int lines = 2 * p128.bk_l;
    cuDoubleComplex ****bk = (cuDoubleComplex****)malloc(p128.n * sizeof(cuDoubleComplex***));
    for (int i = 0; i < p128.n; i++)
    {
        bk[i] = (cuDoubleComplex***)malloc(lines * sizeof(cuDoubleComplex**));
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            bk[i][j] = (cuDoubleComplex**)malloc(2 * sizeof(cuDoubleComplex*));
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                bk[i][j][k] = (cuDoubleComplex*)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
            }
        }
    }
    bkfft(tlwekey, trlwekey, p128, bk);

    // ksk size: (p128.N, p128.ks_t, 1 << p128.ks_basebit, p128.n + 1)
    uint32_t ****ksk = (uint32_t****)malloc(p128.N * sizeof(uint32_t***));
    for (int i = 0; i < p128.N; i++)
    {
        ksk[i] = (uint32_t***)malloc(p128.ks_t * sizeof(uint32_t**));
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            ksk[i][j] = (uint32_t**)malloc((1 << p128.ks_basebit) * sizeof(uint32_t*));
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                ksk[i][j][k] = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
            }
        }    
    }
    kskGen((int32_t*)trlwekey, tlwekey, p128, ksk);


    uint32_t *tlwecipher = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(mutoT(p128.Msize / 2, p128.Msize), tlwekey, p128.n, p128, tlwecipher);

    uint32_t *nc = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    GateBootstrappingFFT(tlwecipher, bk, ksk, p128, nc);
    
    uint32_t msg = tlweSymDec(nc, (int32_t*)trlwekey, p128.n, p128);
    cout << "\nTest_GateBootstrappingFFT decryption result is " << msg << endl;


    free(tlwekey);
    free(trlwekey);
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                free(bk[i][j][k]);
            }
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        for (int j = 0; j < lines; j++)
        {
            free(bk[i][j]);
        }
    }
    for (int i = 0; i < p128.n; i++)
    {
        free(bk[i]);
    }
    free(bk);
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            for (int k = 0; k < (1 << p128.ks_basebit); k++)
            {
                free(ksk[i][j][k]);
            }
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        for (int j = 0; j < p128.ks_t; j++)
        {
            free(ksk[i][j]);
        }    
    }
    for (int i = 0; i < p128.N; i++)
    {
        free(ksk[i]);
    }
    free(ksk);
    free(tlwecipher);
    free(nc);
}






void CircuitBootstrapping(uint32_t *tlwecipher, cuDoubleComplex ****bk, uint32_t ******privksk, Params128 &p128)
{

}
