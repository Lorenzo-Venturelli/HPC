#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "covariance.cuh"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define MEAN_BLOCK_SIZE_X	256
#define MEAN_BLOCK_SIZE_Y	1

#define CENTER_BLOCK_SIZE_X	32
#define CENTER_BLOCK_SIZE_Y	8

#define COVAX_BLOCK_SIZE_X	256
#define COVAX_BLOCK_SIZE_Y	1

#define LARGE_DATASET

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
__host__ static void init_array (int n, double *float_n, double *data)
{
	int i, j;

	*float_n = 1.2;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			data[i * n + j] = ((double) i*j) / M;
		}
	}
}

void covariance(int n, double float_n, double *data, double *symmat, double *mean)
{ 
	int i, j, j1, j2;   

	// Determine mean of column vectors of input data matrix
	for (j = 0; j < n; j = j + 1)
	{
		mean[j] = 0.0;

		for (i = 0; i < n; i = i + 1)
		{
			mean[j] = mean[j] +  data[i * n + j];
		}

		mean[j] = mean[j] / float_n;
	}

	// Center the column vectors
	for (i = 0; i < n; i = i + 1)
	{
		for (j = 0; j < n; j = j + 1)
		{
			data[i * n + j] = data[i * n + j] - mean[j];
		}
	}
	   
	// Calculate the m * m covariance matrix
	#pragma omp parallel num_threads(4)
	{
		#pragma omp for schedule(dynamic, 16)
		for (j1 = 0; j1 < n; j1 = j1 + 1)
		{
			for (j2 = j1; j2 < n; j2 = j2 + 1)
			{
				symmat[j1 * n + j2] = 0.0;
				for (i = 0; i < n; i = i + 1)
				{
					symmat[j1 * n + j2] = symmat[j1 * n + j2] + (data[i * n + j1] * data[i * n + j2]);
				}
				symmat[j2 * n + j1] = symmat[j1 * n + j2];
			}
		}
	}
	
	return;
}

void compare_results(int n, double *symmat_cpu, double *symmat_gpu)
{
	int fail = 0;
	double difference;

	for (int i = 0; i < n; i = i + 1)
	{
		for (int j = 0; j < n; j = j + 1)
		{
			difference = ((symmat_gpu[i * n + j] - symmat_cpu[i * n + j]) / symmat_cpu[i * n + j]) * 100;

			if (difference > 1.05)
			{
				fail = fail + 1;
			}
		}
	}

	fprintf(stderr, "Non-matching CPU - GPU outputs: %d\n", fail);
}

/* Compute the mean vector - STEP 1 */
__global__ void mean_kernel(int n, double float_n, double *data, double *mean)
{
	// Define indexes
	int i, j;

	// Row index to calculate the mean vector
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < n)
	{
		mean[j] = 0.0;
		__syncthreads();

		for(i = 0; i < n; i++)
		{
			atomicadd(&mean[j], data[i * n + j]);
		}
		mean[j] = mean[j] / float_n;
	}

	return;
}

/* Center the data matrix values on the means - STEP 2 */
__global__ void center_kernel(int n, double *data, double *mean)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n)
	{
		data[i * n + j] = data[i * n + j] - mean[j];
	}

	return;
}

/* Calculate the covariance matrix - STEP 3 */
__global__ void covax_kernel(int n, double *data, double *symmat)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j2;

	if (j1 < n)
	{
		for (j2 = j1; j2 < n; j2++)
		{		
			symmat[j1 * n + j2] = 0.0;
			for(i = 0; i < n; i++)
			{
				symmat[j1 * n + j2] += data[i * n + j1] * data[i * n + j2];
			}
			symmat[j2 * n + j1] = symmat[j1 * n + j2];
		}
	}
}


int main(int argc, char *argv[])
{

	double wt;
	struct timespec rt[2];
	int iret = 0;

	/* Retrieve problem size. */
	int n = N;
	fprintf(stderr, "Dataset size: %d\n", N);
 
	/* Variable declaration/allocation. */
	double float_n;
	double *symmat, *mean, *data_d, *mean_d, *symmat_d;
 
	// Allocate CPU structures
	double *data_correct, *symmat_correct, *mean_correct;
	if (argc > 1){
	  n = atoi(argv[1]);
	}
	if (NULL == (data_correct = (double *)malloc(sizeof(*data_correct) * n * n)))
	{
		printf("error: memory allocation for 'x'\n");
		iret = -1;
	}
	if (NULL == (symmat_correct = (double *)malloc(sizeof(*symmat_correct) * n * n)))
	{
		printf("error: memory allocation for 'x'\n");
		iret = -1;
	}
	if (NULL == (mean_correct = (double *)malloc(sizeof(*mean_correct) * n)))
	{
		printf("error: memory allocation for 'x'\n");
		iret = -1;
	}

	// Allocate GPU Host strucutres
	if (NULL == (symmat = (double *)malloc(sizeof(*symmat) * n * n)))
	{
		printf("error: memory allocation for 'x'\n");
		iret = -1;
	}
	if (NULL == (mean = (double *)malloc(sizeof(*mean) * n)))
	{
		printf("error: memory allocation for 'x'\n");
		iret = -1;
	}
	
	// Check errors
	if (0 != iret)
	{
		free(data_correct);
		free(symmat_correct);
		free(mean_correct);

		free(symmat);
		free(mean);
		
		exit(EXIT_FAILURE);
	}   
   
	// Initialize the data structure
	init_array(n, &float_n, data_correct);
  
	/* Allocate CUDA structures */
	gpuErrchk(cudaMalloc((void **)&data_d, sizeof(double) * n * n));
	gpuErrchk(cudaMalloc((void **)&symmat_d, sizeof(double) * n * n));
	gpuErrchk(cudaMalloc((void **)&mean_d, sizeof(double) * n));

	// Copy data structure into GPU Memory
	gpuErrchk(cudaMemcpy(data_d, data_correct, sizeof(double) * n * n, cudaMemcpyHostToDevice));

	// Perfomr the CPU computation of this task
	clock_gettime(CLOCK_REALTIME, rt + 0); 
	covariance(n, float_n, data_correct, symmat_correct, mean_correct);
	clock_gettime(CLOCK_REALTIME, rt + 1);
	wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
	fprintf(stderr, "Covariance (HOST): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

	clock_gettime(CLOCK_REALTIME, rt + 0);
 
	//Calculate Grid and Block sizes for the three kernels
	dim3 mean_dimBlock(MEAN_BLOCK_SIZE_X, MEAN_BLOCK_SIZE_Y);
	dim3 mean_dimGrid((n + MEAN_BLOCK_SIZE_X - 1) / MEAN_BLOCK_SIZE_X, 1);

	dim3 center_dimBlock(CENTER_BLOCK_SIZE_X, CENTER_BLOCK_SIZE_Y);
	dim3 center_dimGrid((n + CENTER_BLOCK_SIZE_X - 1) / CENTER_BLOCK_SIZE_X, (n + CENTER_BLOCK_SIZE_X - 1) / CENTER_BLOCK_SIZE_X);

	dim3 covax_dimBlock(COVAX_BLOCK_SIZE_X, COVAX_BLOCK_SIZE_Y);
	dim3 covax_dimGrid((n + COVAX_BLOCK_SIZE_X - 1) / COVAX_BLOCK_SIZE_X, 1); 
 
	// Run the kernels
	mean_kernel<<<mean_dimGrid, mean_dimBlock>>>(n, float_n, data_d, mean_d);
	center_kernel<<<center_dimGrid, center_dimBlock>>>(n, data_d, mean_d);
	covax_kernel<<<covax_dimGrid, covax_dimBlock>>>(n, data_d, symmat_d);
	cudaDeviceSynchronize();

	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_REALTIME, rt + 1);
	wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
	fprintf(stderr, "Covariance (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

	// Copy back the results
	gpuErrchk(cudaMemcpy(symmat, symmat_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(mean, mean_d, sizeof(double) * n, cudaMemcpyDeviceToHost));
  
	int counter = 0;
	for (int i = 0; i<n; i++)
	{
		if(mean[i] != mean_correct[i]) counter += 1;
	}
	fprintf(stderr, "Mean errors: %d\n", counter);
	
	compare_results(n, symmat_correct, symmat);
	
	gpuErrchk(cudaFree(data_d));
	gpuErrchk(cudaFree(symmat_d));
	gpuErrchk(cudaFree(mean_d));

	free(data_correct);
	free(symmat_correct);
	free(mean_correct);

	free(symmat);
	free(mean);
 
	
	//cudaDeviceReset();

	return 0;
}
