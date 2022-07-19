#include "function2.cuh"

// 以下是采用了多维数组的计算
__global__ void STLU_GPU_1(uint32_t ***database, uint32_t *d_params, uint32_t *d_decbit, int cnt, uint32_t ****t1, uint32_t ***decvec, cuDoubleComplex ***decvecfft)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    // int d0 = int(pow(2, cnt) - 1)+ (int)pow(2, cnt+1) * tid;
    // int d1 = d0 + (int)pow(2, cnt);
    int d0 = (1 << cnt) - 1 + (1 << (cnt + 1)) * tid;
    int d1 = d0 + (1 << cnt);
    // printf("\nd0: %d", d0);
    // printf("\nd1: %d\n", d1);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < d_params[3]; j++)
        {
            database[d1][i][j] = database[d1][i][j] - database[d0][i][j];                           // ok!
            database[d1][i][j] = (uint32_t)((database[d1][i][j] + (uint64_t)d_params[0]) % _two32); // ok!
        }
    }
    // t1 size : threadnum * p128.bk_l * 2 * p128.N
    // printf("tid: %d\n", tid);
    for (int i = 0; i < d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                t1[tid][i][j][k] = database[d1][j][k] >> d_decbit[i];    // ok!
                t1[tid][i][j][k] = t1[tid][i][j][k] & (d_params[1] - 1); // ok!
                t1[tid][i][j][k] = t1[tid][i][j][k] - d_params[1] / 2;   // ok!
            }
        }
    }
    // decvec size: threadnum * (p128.bk_l * 2) * p128.N
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < d_params[2]; i++)
        {
            for (int k = 0; k < d_params[3]; k++)
            {
                // attention!
                decvec[tid][i + j * d_params[2]][k] = t1[tid][i][j][k];
            }
        }
    }
}

__global__ void STLU_GPU_2(uint32_t ***database, cuDoubleComplex ***id, uint32_t *d_params, int cnt, cuDoubleComplex ***decvecfft, cuDoubleComplex ****t4, cuDoubleComplex ***t5)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // i = 0, deal with: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    // i = 1, deal with: (1,3), (5,7), (9,11), (13,15)
    // i = 2, deal with: (3,7), (11,15)
    // i = 3, deal with: (7,15)
    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)

    // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
    int M = d_params[3] / 2;
    for (int i = 0; i < 2 * d_params[2]; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < M; k++)
            {
                t4[tid][i][j][k] = cuCmul(id[i][j][k], decvecfft[tid][i][k]);
            }
        }
    }
    // t5 size: threadnum * 2 * (p128.N / 2)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < M; j++)
        {
            t5[tid][i][j].x = 0;
            t5[tid][i][j].y = 0;
            for (int k = 0; k < 2 * d_params[2]; k++)
            {
                t5[tid][i][j] = cuCadd(t5[tid][i][j], t4[tid][k][i][j]);
            }
        }
    }
}

void Test_STLU_1()
{
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 2, 8, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // 数据库一共有 flen 条数据，每条数据的序号用 dlen 个二进制位来表示
    int dlen = 12;
    int flen = 1 << dlen;
    // attention! host and device should have a backup of database respectively.
    // database size: flen * 2 * p128.N
    uint32_t ***database;
    database = create_3d_array<uint32_t>(flen, 2, p128.N);

    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));

    Database(p128, database, flen, trlwekey);
    // cout << "database: " << endl;
    // for (int i = 0; i < flen; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         for (int k = 0; k < p128.N; k++)
    //         {
    //             cout << database[i][j][k] << ",";
    //         }
    //     }
    //     cout << endl;
    // }
    // cout << "---------------------------------------------" << endl;

    int idnum = 103;
    uint32_t *idbits = (uint32_t *)malloc(dlen * sizeof(uint32_t));
    GetBits(idnum, idbits, dlen);

    uint64_t *mu0 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    uint64_t *mu1 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu0[i] = 0;
        mu1[i] = 0;
    }
    mu1[0] = pow(2, 32);

    int lines = 2 * p128.bk_l;
    // id size : dlen * lines * 2 * (p128.N / 2)
    cuDoubleComplex ****id;
    id = create_4d_array<cuDoubleComplex>(dlen, lines, 2, p128.N / 2);
    for (int i = 0; i < dlen; i++)
    {
        if (idbits[i] == 1)
        {
            trgswfftSymEnc(mu1, trlwekey, p128, id[i]);
        }
        else
        {
            trgswfftSymEnc(mu0, trlwekey, p128, id[i]);
        }
    }

    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    uint32_t *h_params = (uint32_t *)malloc(4 * sizeof(uint32_t));
    h_params[0] = p128.offset;
    h_params[1] = p128.Bg;
    h_params[2] = p128.bk_l;
    h_params[3] = p128.N;
    uint32_t *d_params;
    CHECK(cudaMalloc((void **)&d_params, 4 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_params, h_params, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    uint32_t *d_decbit;
    CHECK(cudaMalloc((void **)&d_decbit, p128.bk_l * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_decbit, p128.decbit, p128.bk_l * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // begin to count time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    int threadnum = flen / 2;
    int M = p128.N / 2;
    for (int cnt = 0; cnt < dlen; cnt++)
    {
        // cout << "\n\ncnt = " << cnt << endl;

        // t1 size: threadnum * p128.bk_l * 2 * p128.N
        uint32_t ****t1;
        t1 = create_4d_array<uint32_t>(threadnum, p128.bk_l, 2, p128.N);
        // decvec size: threadnum * (p128.bk_l * 2) * p128.N
        uint32_t ***decvec;
        decvec = create_3d_array<uint32_t>(threadnum, p128.bk_l * 2, p128.N);
        // decvecfft size: threadnum * (p128.bk_l * 2) * (p128.N / 2)
        cuDoubleComplex ***decvecfft;
        decvecfft = create_3d_array<cuDoubleComplex>(threadnum, p128.bk_l * 2, p128.N / 2);
        // t4 size: threadnum * (2 * p128.bk_l) * 2 * (p128.N / 2)
        cuDoubleComplex ****t4;
        t4 = create_4d_array<cuDoubleComplex>(threadnum, 2 * p128.bk_l, 2, p128.N / 2);
        // t5 size: threadnum * 2 * (p128.N / 2)
        cuDoubleComplex ***t5;
        t5 = create_3d_array<cuDoubleComplex>(threadnum, 2, p128.N / 2);
        // t6 size : threadnum * 2 * p128.N
        uint32_t ***t6;
        t6 = create_3d_array<uint32_t>(threadnum, 2, p128.N);

        // db1 -db0
        // STLU_GPU_1<<<1, threadnum>>>(database, flen, d_params, d_decbit, cnt, t1, decvec, decvecfft);
        if (threadnum > 1024)
        {
            STLU_GPU_1<<<threadnum / 1024, 1024>>>(database, d_params, d_decbit, cnt, t1, decvec, decvecfft);
            cudaDeviceSynchronize();
        }
        else
        {
            // 1, threadnum
            STLU_GPU_1<<<1, threadnum>>>(database, d_params, d_decbit, cnt, t1, decvec, decvecfft);
            cudaDeviceSynchronize();
        }

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2 * p128.bk_l; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    decvecfft[tid][i][j].x = (int32_t)(decvec[tid][i][j]);
                    decvecfft[tid][i][j].y = (int32_t)(decvec[tid][i][j + M]);
                    decvecfft[tid][i][j] = cuCmul(decvecfft[tid][i][j], p128.twist[j]);
                    // cout << decvecfft[tid][i][j].x << "+" << decvecfft[tid][i][j].y << "j, ";
                }
                // cout << endl;
                cufftHandle plan;
                cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
                cufftExecZ2Z(plan, (cuDoubleComplex *)decvecfft[tid][i], (cuDoubleComplex *)decvecfft[tid][i], CUFFT_FORWARD);
                CHECK(cudaDeviceSynchronize());
                cufftDestroy(plan);
            }
        }

        if (threadnum > 1024)
        {
            STLU_GPU_2<<<threadnum / 1024, 1024>>>(database, id[cnt], d_params, cnt, decvecfft, t4, t5);
            cudaDeviceSynchronize();
        }
        else
        {
            STLU_GPU_2<<<1, threadnum>>>(database, id[cnt], d_params, cnt, decvecfft, t4, t5);
            cudaDeviceSynchronize();
        }

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                cufftHandle plan;
                cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
                cufftExecZ2Z(plan, (cuDoubleComplex *)t5[tid][i], (cuDoubleComplex *)t5[tid][i], CUFFT_INVERSE);
                CHECK(cudaDeviceSynchronize());
                cufftDestroy(plan);
            }
        }

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    cuDoubleComplex twist;
                    twist.x = p128.twist[j].x;
                    twist.y = (-1) * p128.twist[j].y;
                    // normalize
                    t5[tid][i][j].x = t5[tid][i][j].x / M;
                    t5[tid][i][j].y = t5[tid][i][j].y / M;
                    t5[tid][i][j] = cuCmul(t5[tid][i][j], twist);
                    t6[tid][i][j] = (uint32_t)t5[tid][i][j].x;
                    t6[tid][i][j + M] = (uint32_t)t5[tid][i][j].y;
                }
            }
        }

        for (int tid = 0; tid < threadnum; tid++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < p128.N; j++)
                {
                    int d0 = (1 << cnt) - 1 + (1 << (cnt + 1)) * tid;
                    int d1 = d0 + (1 << cnt);
                    database[d1][i][j] = ((uint64_t)t6[tid][i][j] + (uint64_t)database[d0][i][j]) % _two32;
                    // cout << database[d1][i][j] << ",";
                }
            }
            // cout << endl;
        }

        // cout << setiosflags(ios::scientific) << setprecision(8);
        // cout << "\ndatabase:\n";
        // for (int i = 0; i < flen; i += 1)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << database[i][j][k] << ",";
        //         }
        //     }
        //     cout << endl;
        // }
        // cout << "database:\n";
        // for (int i = (1 << (cnt + 1)) - 1; i < flen; i += (1 << (cnt + 1)))
        // {
        //     cout << "i: " << i << endl;
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << database[i][j][k] << ",";
        //         }
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nt1:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int t = 0; t < p128.bk_l; t++)
        //     {
        //         for (int j = 0; j < 2; j++)
        //         {
        //             for (int k = 0; k < p128.N; k++)
        //             {
        //                 cout << t1[i][t][j][k] << ",";
        //             }
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\ndecvec:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2*p128.bk_l; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << decvec[i][j][k] << ",";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\ndecvecfft:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2*p128.bk_l; j++)
        //     {
        //         for (int k = 0; k < M; k++)
        //         {
        //             cout << decvecfft[i][j][k].x << "+" << decvecfft[i][j][k].y << "j, ";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\n id:\n";
        // for (int i = 0; i < lines; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N / 2; k++)
        //         {
        //             cout << id[cnt][i][j][k].x << "+" << id[cnt][i][j][k].y << "j, ";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nt4:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int t = 0; t < 2*p128.bk_l; t++)
        //     {
        //         for (int j = 0; j < 2; j++)
        //         {
        //             for (int k = 0; k < p128.N/2; k++)
        //             {
        //                 cout << t4[i][t][j][k].x << "," << t4[i][t][j][k].y << "\t";
        //             }
        //             cout << endl;
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nt5:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N / 2; k++)
        //         {
        //             cout << t5[i][j][k].x << "," << t5[i][j][k].y << "\t";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
        // cout << "\n***************************************" << endl;
        // cout << "\nt6:\n";
        // for (int i = 0; i < threadnum; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         for (int k = 0; k < p128.N; k++)
        //         {
        //             cout << t6[i][j][k] << "\t";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }

        // cout << "\n----------------------------------------------" << endl;
        threadnum = threadnum / 2;

        free_4d_array(t1);
        free_3d_array(decvec);
        free_3d_array(decvecfft);
        free_4d_array(t4);
        free_3d_array(t5);
        free_3d_array(t6);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms.\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(database[flen - 1], trlwekey, p128, msg);
    cout << "\nTable Lookup result is:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    cout << endl;

    free_3d_array(database);
    free(trlwekey);
    free(idbits);
    free(mu0);
    free(mu1);
    free_4d_array(id);
    free(h_params);
    CHECK(cudaFree(d_params));
    CHECK(cudaFree(d_decbit));
    free(msg);
}

// 以下是采用了一维数组的计算
__global__ void TLU_GPU_1(uint32_t *d_database, uint32_t *d_params, uint32_t *d_decbit, int cnt, uint32_t *d_t1)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // d_params[0] = offset, d_params[1] = Bg = 10, d_params[2] = bk_l = 2, d_params[3] = N
    int d0, d1;
    d0 = ((1 << (cnt + 1)) - 2) * d_params[3] + (1 << (cnt + 2)) * d_params[3] * (tid / (2 * d_params[3])) + tid % (2 * d_params[3]);
    d1 = d0 + (1 << (cnt + 1)) * d_params[3];
    // printf("%d: %d, %d\n", tid, d0, d1);
    d_database[d1] = d_database[d1] - d_database[d0];                               // 偶数行 - 奇数行
    d_database[d1] = (uint32_t)((d_database[d1] + (uint64_t)d_params[0]) % _two32); // 每个元素都加上偏移量

    for (int i = 0; i < d_params[2]; i++)
    {
        // idx = 2N * (tid / N) + tid % N + i * N
        int idx = 2 * d_params[3] * (tid / d_params[3]) + tid % d_params[3] + i * d_params[3];
        d_t1[idx] = d_database[d1] >> d_decbit[i]; // d_decbit[0] = 22, d_decbit[1] = 12
        d_t1[idx] = d_t1[idx] & (d_params[1] - 1); // 每个元素 &（Bg - 1)
        d_t1[idx] = d_t1[idx] - d_params[1] / 2;   // 每个元素 - Bg / 2
    }
}

__global__ void TLU_GPU_1(uint32_t *d_params, cuDoubleComplex *d_twist, uint32_t *d_t1, cuDoubleComplex *d_decvecfft)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // 转成复数， 并乘上 twist, twist是一个有N / 2个复数的一维数组
    int M = d_params[3] / 2;
    int index = tid + (tid / M) * M;
    d_decvecfft[tid].x = (int32_t)d_t1[index];
    d_decvecfft[tid].y = (int32_t)d_t1[index + M];
    d_decvecfft[tid] = cuCmul(d_decvecfft[tid], d_twist[index % M]);
}

__global__ void TLU_GPU_2(cuDoubleComplex *id, uint32_t *d_params, cuDoubleComplex *d_decvecfft, cuDoubleComplex *d_t4)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int M = d_params[3] / 2;
    int idsize = 2 * d_params[2] * d_params[3];
    int n = tid / M * d_params[3] + tid % M;
    int idx = (tid / M * d_params[3] + tid % M) % idsize;
    // printf("\n%d, %d", tid, idx);

    d_t4[n] = cuCmul(id[idx], d_decvecfft[tid]); // 乘法
    d_t4[n + M] = cuCmul(id[idx + M], d_decvecfft[tid]);
}

__global__ void TLU_GPU_2(uint32_t *d_params, cuDoubleComplex *d_t4, cuDoubleComplex *d_t5)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idsize = 2 * d_params[2] * d_params[3];

    // 每 2*bk_l=4 行的对位相加，4行-->1行
    int m = tid / d_params[3] * idsize + tid % d_params[3];
    d_t5[tid].x = 0;
    d_t5[tid].y = 0;
    for (int k = 0; k < 2 * d_params[2]; k++)
    {
        d_t5[tid] = cuCadd(d_t5[tid], d_t4[m + k * d_params[3]]);
    }
}

__global__ void TLU_GPU_3(uint32_t *d_params, cuDoubleComplex *d_twist, cuDoubleComplex *d_t5)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int M = d_params[3] / 2;

    d_t5[tid].x = d_t5[tid].x / M;
    d_t5[tid].y = d_t5[tid].y / M;
    d_t5[tid] = cuCmul(d_t5[tid], cuConj(d_twist[tid % M]));
    // __syncthreads();
    // int idx = tid / M * d_params[3] + tid % M;
    // printf("tid: %d, idx: %d, idx+M: %d, d_t5[tid].x: %d, d_t5[tid].y: %d\n", tid, idx, idx+M, (uint32_t)d_t5[tid].x, (uint32_t)d_t5[tid].y);
    // d_t6[idx] = (uint32_t)d_t5[tid].x;
    // d_t6[idx + M] = (uint32_t)d_t5[tid].y;
    // d_t6[idx] = (uint32_t)((int)d_t5[tid].x);
    // d_t6[idx + M] = (uint32_t)((int)d_t5[tid].y);
    // d_t6[idx] = (int)d_t5[tid].x;
    // d_t6[idx + M] = (int)d_t5[tid].y;
}

__global__ void TLU_GPU_3(uint32_t *d_params, cuDoubleComplex *d_t5, uint32_t *d_t6)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int M = d_params[3] / 2;

    int idx = tid / M * d_params[3] + tid % M;
    // printf("tid: %d, idx: %d, idx+M: %d, d_t5[tid].x: %d, d_t5[tid].y: %d\n", tid, idx, idx+M, (uint32_t)d_t5[tid].x, (uint32_t)d_t5[tid].y);
    // 这里的类型转换需要注意，不是直接转成 uint，因为会出错，我也不知道为什么
    d_t6[idx] = __double2ll_ru(d_t5[tid].x);
    d_t6[idx + M] = __double2ll_ru(d_t5[tid].y);
}

__global__ void TLU_GPU_4(uint32_t *d_database, uint32_t *d_params, uint32_t *d_t6, int cnt)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int d0, d1;
    d0 = ((1 << (cnt + 1)) - 2) * d_params[3] + (1 << (cnt + 2)) * d_params[3] * (tid / (2 * d_params[3])) + tid % (2 * d_params[3]);
    d1 = d0 + (1 << (cnt + 1)) * d_params[3];
    // 最后计算的结果要加上奇数行
    d_database[d1] = ((uint64_t)d_t6[tid] + (uint64_t)d_database[d0]) % _two32;
}

// 在 CPU 分配空间的变量会用 h 开头，在 GPU 分配空间的变量用 d 开头
// cudaMemcpy 操作都是为了检验数据计算的正确性
// 核函数的调用都会进行判断，如果需要的线程数少于1024，则<<<1, threadnum>>> ; 如果超过1024，网格大小相应增加

float Test_STLU_2()
{
    // Params128 第一个参数就是 N = 128
    Params128 p128 = Params128(1024, 630, 2048, 2, 10, 4, 9, 2, 8, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 2);
    // 数据库一共有 flen 条数据，每条数据的序号用 dlen 个二进制位来表示
    int dlen = 13;
    int flen = 1 << dlen;
    int M = p128.N / 2;

    // 每条数据大小是 2 * N，所以 database size: flen * 2 * p128.N
    uint32_t *h_database, *d_database;
    h_database = (uint32_t *)malloc(flen * 2 * p128.N * sizeof(uint32_t));
    CHECK(cudaMalloc((void **)&d_database, flen * 2 * p128.N * sizeof(uint32_t)));

    // trlwekey 是密钥，DB_generation 生成数据库，第i条数据对应的数字i的密文
    uint32_t *trlwekey = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    DB_generation(p128, h_database, flen, trlwekey);
    CHECK(cudaMemcpy(d_database, h_database, flen * 2 * p128.N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // cout << "database: " << endl;
    // for (int i = 0; i < flen; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         for (int k = 0; k < p128.N; k++)
    //         {
    //             cout << h_database[i * 2 * p128.N + j * p128.N + k] << ",";
    //         }
    //     }
    //     cout << endl;
    // }
    // cout << "---------------------------------------------" << endl;

    // 要查找第 idnum = 35 条数据， GetBits 把 idnum 转成二进制并逆置，得到 idbits =[1, 1, 0, 0, 0, 1]
    int idnum = 118;
    uint32_t *idbits = (uint32_t *)malloc(dlen * sizeof(uint32_t));
    GetBits(idnum, idbits, dlen);

    // 下面是把 idbits 的每一个比特转成一个 lines * 2 * (N / 2) 的三维密文，lines 在这里通常都设置成 bk_l 的2倍，也就是4
    // idbits 一共有dlen位，每位比特都转成 lines * 2 * (N / 2) 的密文，所以 idsize = dlen * lines * 2 * (p128.N / 2)
    uint64_t *mu0 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    uint64_t *mu1 = (uint64_t *)malloc(p128.N * sizeof(uint64_t));
    for (int i = 0; i < p128.N; i++)
    {
        mu0[i] = 0;
        mu1[i] = 0;
    }
    mu1[0] = pow(2, 32);

    // bk_l 通常都设置成 2, 所以一个密文
    int lines = 2 * p128.bk_l;
    int idsize = lines * p128.N;
    // id size : dlen * lines * 2 * (p128.N / 2)
    cuDoubleComplex *h_id = (cuDoubleComplex *)malloc(dlen * lines * p128.N * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_id;
    CHECK(cudaMalloc((void **)&d_id, dlen * lines * p128.N * sizeof(cuDoubleComplex)));
    cuDoubleComplex ***trgsw = (cuDoubleComplex ***)malloc(lines * sizeof(cuDoubleComplex **));
    for (int i = 0; i < lines; i++)
    {
        trgsw[i] = (cuDoubleComplex **)malloc(2 * sizeof(cuDoubleComplex *));
    }
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            trgsw[i][j] = (cuDoubleComplex *)malloc(p128.N / 2 * sizeof(cuDoubleComplex));
        }
    }

    cout << setiosflags(ios::scientific) << setprecision(8);
    for (int i = 0; i < dlen; i++)
    {
        if (idbits[i] == 1)
        {
            trgswfftSymEnc(mu1, trlwekey, p128, trgsw);
        }
        else
        {
            trgswfftSymEnc(mu0, trlwekey, p128, trgsw);
        }
        for (int j = 0; j < lines; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                for (int l = 0; l < p128.N / 2; l++)
                {
                    int idx = i * lines * p128.N + j * p128.N + k * p128.N / 2 + l;
                    h_id[idx] = trgsw[j][k][l];
                    // int index = j * p128.N + k * p128.N / 2 + l;
                    // id[i][index] = trgsw[j][k][l];
                    // cout << h_id[idx].x << "+" << h_id[idx].y << "j, ";
                }
            }
            // cout << endl;
        }
    }
    CHECK(cudaMemcpy(d_id, h_id, dlen * lines * p128.N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // params=np.array([p128.offset,p128.Bg,p128.bk_l, p128.N],dtype=np.uint32)
    uint32_t *h_params = (uint32_t *)malloc(4 * sizeof(uint32_t));
    h_params[0] = p128.offset;
    h_params[1] = p128.Bg;
    h_params[2] = p128.bk_l;
    h_params[3] = p128.N;
    uint32_t *d_params;
    CHECK(cudaMalloc((void **)&d_params, 4 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_params, h_params, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // d_decbit[0] = 22, d_decbit[1] = 12
    uint32_t *d_decbit;
    CHECK(cudaMalloc((void **)&d_decbit, p128.bk_l * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_decbit, p128.decbit, p128.bk_l * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // d_twsit 是一个有 N / 2 个复数的一维数组
    cuDoubleComplex *d_twist;
    CHECK(cudaMalloc((void **)&d_twist, M * sizeof(cuDoubleComplex)));
    CHECK(cudaMemcpy(d_twist, p128.twist, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // begin to count time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    // 查表一共需要 dlen 次，所以循环 dlen 次
    int threadnum = flen * p128.N;
    for (int cnt = 0; cnt < dlen; cnt++)
    {
        // cout << "\n\ncnt = " << cnt << endl;

        uint32_t *h_t1 = (uint32_t *)malloc(threadnum * 2 * sizeof(uint32_t));
        uint32_t *d_t1;
        CHECK(cudaMalloc((void **)&d_t1, threadnum * 2 * sizeof(uint32_t)));
        cuDoubleComplex *h_decvecfft = (cuDoubleComplex *)malloc(threadnum * sizeof(cuDoubleComplex));
        cuDoubleComplex *d_decvecfft;
        CHECK(cudaMalloc((void **)&d_decvecfft, threadnum * sizeof(cuDoubleComplex)));
        cuDoubleComplex *h_t4 = (cuDoubleComplex *)malloc(2 * threadnum * sizeof(cuDoubleComplex));
        cuDoubleComplex *d_t4;
        CHECK(cudaMalloc((void **)&d_t4, 2 * threadnum * sizeof(cuDoubleComplex)));
        cuDoubleComplex *h_t5 = (cuDoubleComplex *)malloc(threadnum / 2 * sizeof(cuDoubleComplex));
        cuDoubleComplex *d_t5;
        CHECK(cudaMalloc((void **)&d_t5, threadnum / 2 * sizeof(cuDoubleComplex)));
        uint32_t *h_t6 = (uint32_t *)malloc(threadnum * sizeof(uint32_t));
        uint32_t *d_t6;
        CHECK(cudaMalloc((void **)&d_t6, threadnum * sizeof(uint32_t)));

        if (threadnum > 1024)
        {
            TLU_GPU_1<<<threadnum / 1024, 1024>>>(d_database, d_params, d_decbit, cnt, d_t1);
            cudaDeviceSynchronize();
            TLU_GPU_1<<<threadnum / 1024, 1024>>>(d_params, d_twist, d_t1, d_decvecfft);
            cudaDeviceSynchronize();
        }
        else
        {
            TLU_GPU_1<<<1, threadnum>>>(d_database, d_params, d_decbit, cnt, d_t1);
            cudaDeviceSynchronize();
            TLU_GPU_1<<<1, threadnum>>>(d_params, d_twist, d_t1, d_decvecfft);
            cudaDeviceSynchronize();
        }
        // CHECK(cudaMemcpy(h_database, d_database, flen * 2 * p128.N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        // CHECK(cudaMemcpy(h_t1, d_t1, threadnum * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // 这一部分是想要实现一维数组批量FFT，每M个数据进行一维FFT，但是计算结果有时不正确
        // d_decvecfft size: threadnum = flen * N
        int NX = M;
        int NY = threadnum / M;
        cufftHandle plan;
        int rank = 1;
        int n[1];
        n[0] = NX;
        int istride = 1;
        int idist = NX;
        int ostride = 1;
        int odist = NX;
        int inembed[2];
        int onembed[2];
        inembed[0] = NX;
        onembed[0] = NX;
        inembed[1] = NY;
        onembed[1] = NX;
        cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, NY);
        cufftExecZ2Z(plan, d_decvecfft, d_decvecfft, CUFFT_FORWARD);
        cufftDestroy(plan);
        // cudaMemcpy(h_decvecfft, d_decvecfft, threadnum * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        // for (int i = 0; i < threadnum / M; i++)
        // {
        //     cufftHandle plan;
        //     cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
        //     cufftExecZ2Z(plan, (cuDoubleComplex *)(d_decvecfft+i*M), (cuDoubleComplex *)(d_decvecfft+i*M), CUFFT_FORWARD);
        //     CHECK(cudaDeviceSynchronize());
        // }
        // CHECK(cudaMemcpy(h_decvecfft, d_decvecfft, threadnum * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        if (threadnum > 1024)
        {
            TLU_GPU_2<<<threadnum / 1024, 1024>>>(d_id + cnt * idsize, d_params, d_decvecfft, d_t4);
            cudaDeviceSynchronize();
        }
        else
        {
            TLU_GPU_2<<<1, threadnum>>>(d_id + cnt * idsize, d_params, d_decvecfft, d_t4);
            cudaDeviceSynchronize();
        }
        if (threadnum / 2 > 1024)
        {
            TLU_GPU_2<<<threadnum / 2048, 1024>>>(d_params, d_t4, d_t5);
            cudaDeviceSynchronize();
        }
        else
        {
            TLU_GPU_2<<<1, threadnum / 2>>>(d_params, d_t4, d_t5);
            cudaDeviceSynchronize();
        }
        // CHECK(cudaMemcpy(h_t4, d_t4, threadnum * 2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // CHECK(cudaMemcpy(h_t5, d_t5, threadnum / 2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // FFT逆变换
        NX = M;
        NY = threadnum / p128.N;
        rank = 1;
        n[0] = NX;
        istride = 1;
        idist = NX;
        ostride = 1;
        odist = NX;
        inembed[0] = NX;
        onembed[0] = NX;
        inembed[1] = NY;
        onembed[1] = NX;
        cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, NY);
        cufftExecZ2Z(plan, d_t5, d_t5, CUFFT_INVERSE);
        cufftDestroy(plan);
        // for (int i = 0; i < threadnum / p128.N; i++)
        // {
        //     cufftHandle plan;
        //     cufftPlan1d(&plan, M, CUFFT_Z2Z, 1);
        //     cufftExecZ2Z(plan, (cuDoubleComplex *)(d_t5+i*M), (cuDoubleComplex *)(d_t5+i*M), CUFFT_INVERSE);
        //     CHECK(cudaDeviceSynchronize());
        // }
        if (threadnum / 2 > 1024)
        {
            TLU_GPU_3<<<threadnum / 2048, 1024>>>(d_params, d_twist, d_t5);
            CHECK(cudaDeviceSynchronize());
            TLU_GPU_3<<<threadnum / 2048, 1024>>>(d_params, d_t5, d_t6);
            CHECK(cudaDeviceSynchronize());
        }
        else
        {
            TLU_GPU_3<<<1, threadnum / 2>>>(d_params, d_twist, d_t5);
            CHECK(cudaDeviceSynchronize());
            TLU_GPU_3<<<1, threadnum / 2>>>(d_params, d_t5, d_t6);
            CHECK(cudaDeviceSynchronize());
        }
        // // CHECK(cudaMemcpy(h_t5, d_t5, threadnum / 2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // // // 把复数的实部和虚部抽取出来，这一步原本要放在核函数，但是在核函数内部，double --> uint32_t 有问题
        // // for (int tid = 0; tid < threadnum / 2; tid++)
        // // {
        // //     int idx = tid / M * p128.N + tid % M;
        // //     h_t6[idx] = (uint32_t)h_t5[tid].x;
        // //     h_t6[idx+M] = (uint32_t)h_t5[tid].y;
        // // }
        // // CHECK(cudaMemcpy(d_t6, h_t6, threadnum * sizeof(uint32_t), cudaMemcpyHostToDevice));

        if (threadnum > 1024)
        {
            TLU_GPU_4<<<threadnum / 1024, 1024>>>(d_database, d_params, d_t6, cnt);
            CHECK(cudaDeviceSynchronize());
        }
        else
        {
            TLU_GPU_4<<<1, threadnum>>>(d_database, d_params, d_t6, cnt);
            CHECK(cudaDeviceSynchronize());
        }

        // // 以下注释掉的部分都是为了检验中间结果
        // cout << "\n database:" << endl;
        // for (int tid = 0; tid < threadnum; tid++)
        // {
        //     int d0, d1;
        //     d0 = ((1 << (cnt+1)) - 2) * p128.N + (1 << (cnt+2)) * p128.N * (tid / (2 * p128.N)) + tid % (2 * p128.N);
        //     d1 = d0 + (1 << (cnt+1)) * p128.N;
        //     if(tid % (2 * p128.N) == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_database[d1] << ", ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n t1:" << endl;
        // for (int i = 0; i < threadnum * 2; i++)
        // {
        //     if(i % (2 * p128.N) == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_t1[i] << ", ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n decvecfft:" << endl;
        // for (int i = 0; i < threadnum; i++)
        // {
        //     if(i % p128.N == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_decvecfft[i].x << "+" << h_decvecfft[i].y << "j    ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n id:" << endl;
        // for (int i = cnt * idsize; i < (cnt+1) * idsize; i++)
        // {
        //     if(i % p128.N == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_id[i].x << "+" << h_id[i].y << "j, ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n t4:" << endl;
        // for (int i = 0; i < 2 * threadnum; i++)
        // {
        //     if(i % p128.N == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_t4[i].x << "+" << h_t4[i].y << "j, ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n t5:" << endl;
        // for (int i = 0; i < threadnum / 2; i++)
        // {
        //     if(i % p128.N == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_t5[i].x << "+" << h_t5[i].y << "j, ";
        // }
        // cout << "\n\n****************************************************" << endl;
        // cout << "\n t6:" << endl;
        // for (int i = 0; i < threadnum; i++)
        // {
        //     if(i % p128.N == 0)
        //     {
        //         cout << endl;
        //     }
        //     cout << h_t6[i] << "  ";
        // }
        // cout << "\n----------------------------------------------" << endl;

        threadnum = threadnum / 2;

        CHECK(cudaFree(d_t1));
        free(h_t1);
        CHECK(cudaFree(d_decvecfft));
        free(h_decvecfft);
        CHECK(cudaFree(d_t4));
        free(h_t4);
        CHECK(cudaFree(d_t5));
        free(h_t5);
        CHECK(cudaFree(d_t6));
        free(h_t6);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms.\n", elapsed_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 这里其实只需要把最后的 2 * N 个数据传过来即可，然后放到二维数组 c 中，调用trlweSymDec进行解密
    CHECK(cudaMemcpy(h_database, d_database, flen * 2 * p128.N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t **c = (uint32_t **)malloc(sizeof(uint32_t *) * 2);
    for (int i = 0; i < 2; i++)
    {
        c[i] = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.N; j++)
        {
            c[i][j] = h_database[(flen - 1) * 2 * p128.N + i * p128.N + j];
        }
    }

    uint32_t *msg = (uint32_t *)malloc(sizeof(uint32_t) * p128.N);
    trlweSymDec(c, trlwekey, p128, msg);
    cout << "\nTable Lookup result is:\n";
    for (int i = 0; i < p128.N; i++)
    {
        cout << msg[i] << " ";
    }
    cout << endl;

    free(h_database);
    CHECK(cudaFree(d_database));
    free(trlwekey);
    free(idbits);
    free(mu0);
    free(mu1);
    free(h_id);
    CHECK(cudaFree(d_id));
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
    free(h_params);
    CHECK(cudaFree(d_params));
    CHECK(cudaFree(d_decbit));
    CHECK(cudaFree(d_twist));
    for (int i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
    free(msg);

    return elapsed_time;
}


void Test_CircuitBootstrapping()
{
    Params128 p128 = Params128(4, 2, 2048, 2, 10, 4, 9, 8, 2, 10, 3, pow(2.0, -15.4), pow(2.0, -28), pow(2.0, -31), pow(2.0, -44), 8);
   
    int32_t *tlwekey = (int32_t *)malloc(p128.n * sizeof(int32_t));
    uint64_t *trlwekey = (uint64_t*)malloc(p128.nbar * sizeof(uint64_t));
    uint32_t *trlwekeyN = (uint32_t *)malloc(p128.N * sizeof(uint32_t));
    tlweKeyGen(tlwekey, p128.n);
    trlweKeyGen64(trlwekey, p128.nbar);
    trlweKeyGen(trlwekeyN, p128.N);
    
    // bk size: p128.n * (2 * p128.bk_lbar) * 2 * (p128.nbar / 2)     (630,4,2,512)
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

    // privksk size :(2, nbar + 1, p128.ks_tbar, 1 << p128.ks_basebitbar , 2, N)
    uint32_t ******privksk = (uint32_t******)malloc(2 * sizeof(uint32_t*****));
    for (int i = 0; i < 2; i++)
    {
        privksk[i] = (uint32_t*****)malloc((p128.nbar + 1) * sizeof(uint32_t****));
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            privksk[i][j] = (uint32_t****)malloc(p128.ks_tbar * sizeof(uint32_t***));
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                privksk[i][j][k] = (uint32_t***)malloc((1 << p128.ks_basebitbar) * sizeof(uint32_t**));
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebitbar); l++)
                {
                    privksk[i][j][k][l] = (uint32_t**)malloc(2 * sizeof(uint32_t*));
                }    
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebitbar); l++)
                {
                    for (int m = 0; m < 2; m++)
                    {
                        privksk[i][j][k][l][m] = (uint32_t*)malloc(p128.N * sizeof(uint32_t));
                    }    
                }    
            }    
        }    
    }
    privkskGen2(trlwekey, trlwekeyN, p128, privksk);

    uint32_t a = 1;
    uint32_t *ca = (uint32_t*)malloc((p128.n + 1) * sizeof(uint32_t));
    tlweSymEnc(dtot32((double)1 * a /2), tlwekey, p128.n, p128, ca);



    

    free(tlwekey);
    free(trlwekey);
    free(trlwekeyN);
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
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebitbar); l++)
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
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                for (int l = 0; l < (1 << p128.ks_basebitbar); l++)
                {
                    free(privksk[i][j][k][l]);
                }    
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            for (int k = 0; k < p128.ks_tbar; k++)
            {
                free(privksk[i][j][k]);
            }    
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < p128.nbar + 1; j++)
        {
            free(privksk[i][j]);
        }    
    }
    for (int i = 0; i < 2; i++)
    {
        free(privksk[i]);
    }
    free(privksk);
    free(ca);
}



int main()
{
    // float average = 0.0;
    // int cnt = 101;
    // for (int i = 0; i < cnt; i++)
    // {
    //     cout << "\n\nTest " << i + 1 << endl << endl;
    //     if(i == 0)
    //     {
    //         Test_STLU_2();
    //     }
    //     else{
    //         average += Test_STLU_2();
    //     }
    //     // Test_STLU_2();
    // }
    // printf("Average Time = %g ms.\n", average / (cnt-1));

    // Test_STLU_2();
    // Test_Sampleextract();
    // Test_TLWE();
    // Test_TRLWE();
    // Test_TRLWE64();
    // Test_TRGSW();
    // Test_TRGSW64();
    // Test_ExternalProduct();
    // Test_ExternalProduct64();
    // Test_CMUXFFT();
    // Test_CMUXFFT64();
    // Test_STLU_1();
    // Test_IdentityKeySwitch();
    // Test_BlindRotateFFT();
    Test_BlindRotateFFT64();
    // Test_TLWE2TRLWE();
    // Test_PubKS();
    // Test_GateBootstrappingTLWE2TRLWEFFT();
    // Test_GateBootstrappingFFT();
    // Test_CircuitBootstrapping();
    // printf("%f", pow(2, 64) * 1234.457);
    // double a = 940573869575210495;
    // double b = 9241421688455823360;
    // double c = uint64_t(b);
    // cout << "b: " << b << endl;
    // cout << "c: " << c << endl;
    // cout << DoubleToUint64(a + b) << endl;
    // uint64_t d = 15496291841009913702;
    // uint64_t e = 9241421688455823360;
    // cout << double(d) << endl;
    // int64_t e = (int64_t)(fmod(double(d), _two64));
    // cout << "e: " << e << endl;
    // cout << (int64_t)(e + _two64) << endl;


    return 0;
}