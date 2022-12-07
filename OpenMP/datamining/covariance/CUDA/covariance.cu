#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ double atomicadd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* Array initialization. */
__host__ static void init_array (int m, int n, double *float_n, double *data)
{
	int i, j;

	*float_n = 1.2;

	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			data[i * m + j] = ((double) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
	 Can be used also to check the correctness of the output. */
__host__ static void print_array(int m, double *symmat, double *correct)

{
	int i, j;
	int ok = 0;

	for (i = 0; i<m; i++)
		for (j = 0; j < m; j++) {
			fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i * m + j]);
			if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
			if (correct[i * m + j] != symmat[i * m + j]) {ok++;}
		}
	fprintf (stderr, "\n");
	printf("Corretto: %d", ok); 
}


/* Main computational kernel. The whole function will be timed,
	 including the call and return. */
__global__ static void kernel_covariance(
	int m, 
	int n, 
	double float_n, 
	double *data, 
	double *symmat, 
	double *correct,
	double *mean
	)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;

	if(row > m || column > n) {
		return;
	}

	/* Determine mean of column vectors of input data matrix */ //[row * m + column]
	mean[row] = 0.0;
	mean[row] = mean[row] + data[column * n + row];
	mean[row] = mean[row] / float_n;

	/*for (j = 0; j < m; j++)
	{
		mean[j] = 0.0;

		for (i = 0; i < n; i++)
			mean[j] += data[i][j];

		mean[j] /= float_n;

	//for (k = _PB_N-1; k >=0; k--)
	//	data[j][k] -= mean[j];
	}*/
			
	/* Center the column vectors. */
	data[column * n + row] = data[column * n + row] - mean[row];
	__syncthreads();

	/*
	for (i = 0; i < n; i++)    
		for (j = 0; j < m; j++)
			data[i][j] -= mean[j];
	*/

			
	/* Calculate the m * m covariance matrix. */
	for(int j = row; j < m; j = j + 1)
	{
		symmat[row * m + j] = 0.0;
		atomicadd(&symmat[row * m + j], symmat[row * m + j] + data[column * n + row] * data[column * n + j]);
		symmat[j * m + row] = symmat[row * m + j];
	}

/*
	for (j1 = 0; j1 < m; j1++)
	{
		for (j2 = j1; j2 < m; j2++)
		{
			symmat[j1][j2] = 0.0;
			for (i = 0; i < n; i++)
			{
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
			symmat[j2][j1] = symmat[j1][j2];
		}
	}
	*/
	return;
}

__host__ int main(int argc, char** argv)
{
	double wt;
	struct timespec rt[2];

	/* Retrieve problem size. */
	int n = N;
	int m = M;

	/* Variable declaration/allocation. */
	double float_n;
	double *data, *symmat, *correct, *mean;

	/* Allocate CUDA structures */
	gpuErrchk(cudaMallocManaged((void **)&data, sizeof(double) * N * M));
	gpuErrchk(cudaMallocManaged((void **)&symmat, sizeof(double) * M * M));
	gpuErrchk(cudaMallocManaged((void **)&correct, sizeof(double) * M * M));
	gpuErrchk(cudaMallocManaged((void **)&mean, sizeof(double) * M));

	// Calculate Block size
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

	/* Initialize array(s). */
	init_array(m, n, &float_n, data);

	/* Run kernel. */
	clock_gettime(CLOCK_REALTIME, rt + 0);
	kernel_covariance<<<dimGrid, dimBlock>>>(m, n, float_n, data, symmat, correct, mean);
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_REALTIME, rt + 1);

	wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

	gpuErrchk(cudaFree(data));
	gpuErrchk(cudaFree(symmat));
	gpuErrchk(cudaFree(correct));
	gpuErrchk(cudaFree(mean));

	//cudaDeviceReset();

	//kernel_covariance(m, n, float_n, data, symmat, correct, mean);

	return 0;
}
