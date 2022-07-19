 
#include <iostream>
#include <assert.h>
using namespace std;

template<typename T>
T** create_2d_array(int a, int b) {
	T *base;
	cudaError_t err = cudaMallocManaged(&base, a*b * sizeof(T));
	assert(err == cudaSuccess);
	T **ary;
	err = cudaMallocManaged(&ary, a * sizeof(T*));
	assert(err == cudaSuccess);
	for (int i = 0; i < a; i++) {
		ary[i] = base + i*b;
	}
	return ary;
}

template<typename T>
T*** create_3d_array(int a, int b, int c) {
	T *base;
	cudaError_t err = cudaMallocManaged(&base, a*b*c * sizeof(T));
	assert(err == cudaSuccess);
	T ***ary;
	err = cudaMallocManaged(&ary, (a + a * b) * sizeof(T*));
	assert(err == cudaSuccess);
	for (int i = 0; i < a; i++) {
		ary[i] = (T **)((ary + a) + i * b);
		for (int j = 0; j < b; j++) {
			ary[i][j] = base + (i*b + j)*c;
		}
	}
	return ary;
}

template<typename T>
T**** create_4d_array(int a, int b, int c, int d) {
	T *base;
	cudaError_t err = cudaMallocManaged(&base, a*b*c*d * sizeof(T));
	assert(err == cudaSuccess);
	T ****ary;
	err = cudaMallocManaged(&ary, (a + a * b + a * b*c) * sizeof(T*));
	assert(err == cudaSuccess);
	for (int i = 0; i < a; i++) {
		ary[i] = (T ***)((ary + a) + i * b);
		for (int j = 0; j < b; j++) {
			ary[i][j] = (T **)((ary + a + a * b) + i * b*c + j * c);
			for (int k = 0; k < c; k++)
				ary[i][j][k] = base + ((i*b + j)*c + k)*d;
		}
	}
	return ary;
}
 
template<typename T>
void free_4d_array(T**** ary) {
	if (ary[0][0][0]) cudaFree(ary[0][0][0]);
	if (ary) cudaFree(ary);
}

template<typename T>
void free_3d_array(T*** ary) {
	if (ary[0][0]) cudaFree(ary[0][0]);
	if (ary) cudaFree(ary);
}

template<typename T>
void free_2d_array(T** ary) {
	if (ary[0]) cudaFree(ary[0]);
	if (ary) cudaFree(ary);
}
 
 
template<typename T>
__global__ void fill(T**** data, int a, int b, int c, int d) {
	uint32_t val = 0;
	for (int i = 0; i < a; i++)
		for (int j = 0; j < b; j++)
			for (int k = 0; k < c; k++)
				for (int l = 0; l < d; l++)
					data[i][j][k][l] = val++;
}

template<typename T>
__global__ void fill2(T*** data, int a, int b, int c) {
	uint32_t val = 0;
	for (int i = 0; i < a; i++)
		for (int j = 0; j < b; j++)
			for (int k = 0; k < c; k++)
					data[i][j][k] = val++;
}

template<typename T>
__global__ void fill3(T** data, int a, int b) {
	uint32_t val = 0;
	for (int i = 0; i < a; i++)
		for (int j = 0; j < b; j++)
			data[i][j] = val++;
}
 
void report_gpu_mem()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << "Free = " << free << " Total = " << total << std::endl;
}
 
// int main() {
// 	report_gpu_mem();
 
// 	uint32_t **data2;
// 	std::cout << "allocating..." << std::endl;
// 	data2 = create_2d_array<uint32_t>(4, 3);
 
// 	report_gpu_mem();
 
// 	fill3 << <1, 1 >> > (data2, 4, 3);
// 	cudaError_t err = cudaDeviceSynchronize();
// 	assert(err == cudaSuccess);
 
// 	std::cout << "validating..." << std::endl;
// 	for (int i = 0; i < 4 * 3; i++)
// 		if (*(data2[0] + i) != i) { std::cout << "mismatch at " << i << " was " << *(data2[0] + i) << std::endl; return -1; }


//     for (int i = 0; i < 4; i++)
//     {
//         for (int j = 0; j < 3; j++)
//         {
// 			cout << data2[i][j] << "\t";
//         }
//         cout << endl;
//     }
    
// 	free_2d_array(data2);
// 	return 0;
// }