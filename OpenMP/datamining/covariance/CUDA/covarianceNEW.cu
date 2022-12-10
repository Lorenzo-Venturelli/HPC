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

void covariance(
	int m, 
	int n, 
	double *float_n, 
	double *data_correct, 
	double *symmat_correct, 
	double *mean_correct
	)
{ 
  
  int i, j, j1, j2;
	*float_n = 1.2;    
   
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++){
   
			data_correct[i * m + j] = ((double)i*j) / m;
      }
   }  

	for (j = 0; j < m; j++)
	{
		mean_correct[j] = 0.0;

		for (i = 0; i < n; i++)
			mean_correct[j] += data_correct[i * m + j];

		mean_correct[j] /= *float_n;
	}
 
	for (i = 0; i < n; i++)    
		for (j = 0; j < m; j++)
			data_correct[i * n + j] -= mean_correct[j];
       

#pragma omp parallel num_threads(4)
{
  #pragma omp for schedule(dynamic, 16)
	for (j1 = 0; j1 < m; j1++)
	{
		for (j2 = j1; j2 < m; j2++)
		{
			symmat_correct[j1 * m + j2] = 0.0;
			for (i = 0; i < n; i++)
			{
				symmat_correct[j1 * m + j2] += data_correct[i * m + j1] * data_correct[i * m + j2];
			}
			symmat_correct[j2 * m + j1] = symmat_correct[j1 * m + j2];
		}
	}
 }
 
}


/* Main computational kernel. The whole function will be timed,
	 including the call and return. */
__global__ void kernel_covariance(
	int m, 
	int n, 
	double float_n, 
	double *d_data, 
	double *d_symmat,
	double *d_mean
	)
{
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int column = threadIdx.y + blockIdx.y * blockDim.y;
 
  d_mean[column] = 0;
	if(row < m && column < n) {
 
    atomicadd(&d_mean[column], d_data[row * n + column]);
  	//d_mean[column] = d_mean[column] + d_data[row * n + column];  	
    d_mean[column] = d_mean[column] / float_n;  
    
    d_data[row * n + column] -= d_mean[column];
    
    /*for(int j = row; j < m; j = j + 1){
		  d_symmat[row * m + j] = 0.0;
		  atomicadd(&d_symmat[row * m + j], d_symmat[row * m + j] + d_data[column * n + row] * d_data[column * n + j]);
		  //symmat[j * m + row] = symmat[row * m + j];
	  }*/
     
     //__syncthreads();
     
   double sum = 0.0;    
   if(column>=row){
      d_symmat[row * m + column] = 0.0;           
      for(int j = 0; j < m; j++){           
        sum +=  d_data[j * m + row] * d_data[j * n + column];
	    }
      atomicadd(&d_symmat[row * m + column], sum);      
      d_symmat[column * m + row] = d_symmat[row * m + column];
    }    
	}
 
  //__syncthreads();
  
	/* Determine mean of column vectors of input data matrix */ //[row * m + column]
	

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
	//d_data[column * n + row] = d_data[column * n + row] - d_mean[column];
	//__syncthreads();

	/*
	for (i = 0; i < n; i++)    
		for (j = 0; j < m; j++)
			data[i][j] -= mean[j];
	*/

			
	/* Calculate the m * m covariance matrix. */
	//for(int j = row; j < m; j = j + 1)
	//{
		//symmat[row * m + j] = 0.0;
		//atomicadd(&symmat[row * m + j], symmat[row * m + j] + data[column * n + row] * data[column * n + j]);
		//symmat[j * m + row] = symmat[row * m + j];
	//}
 
 
 //for (int i = 0; i<m*n; i++){
   //   printf("%.2f\n", symmat[i]);
      //if (correct[i][j] != symmat[i][j]) {ok++;}
    //}

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
	//return;
}

int main(int argc, char *argv[])
{

	double wt;
	struct timespec rt[2];
  int iret = 0;

	/* Retrieve problem size. */
	int n = N;
	int m = M; 
 
	/* Variable declaration/allocation. */
	double float_n;
	double *data, *symmat, *mean;
 
  double *data_correct, *symmat_correct, *mean_correct;
    if (argc > 1){
      n = atoi(argv[1]);
    }
    if (NULL == (data_correct = (double *)malloc(sizeof(*data_correct) * N * N)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (symmat_correct = (double *)malloc(sizeof(*symmat_correct) * n * m)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (mean_correct = (double *)malloc(sizeof(*mean_correct) * m)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (data = (double *)malloc(sizeof(*data) * N * N)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (symmat = (double *)malloc(sizeof(*symmat) * n * m)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (mean = (double *)malloc(sizeof(*mean) * m)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    
  if (0 != iret)
    {
        free(data_correct);
        free(symmat_correct);
        free(mean_correct);
        free(data);
        free(symmat);
        free(mean);
        
        exit(EXIT_FAILURE);
    }   
   
  clock_gettime(CLOCK_REALTIME, rt + 0); 
  covariance(m, n, &float_n, data_correct, symmat_correct, mean_correct);
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("Covariance (HOST): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));
  
	/* Allocate CUDA structures */
	gpuErrchk(cudaMallocManaged((void **)&data, sizeof(double) * N * M));
	//gpuErrchk(cudaMallocManaged((void **)&symmat, sizeof(double) * M * M));
	//gpuErrchk(cudaMallocManaged((void **)&correct, sizeof(double) * M * M));
	//gpuErrchk(cudaMallocManaged((void **)&mean, sizeof(double) * n));
    
	/* Initialize array(s). */
	init_array(m, n, &float_n, data);
 
  double *d_symmat, *d_mean, *d_data;
  gpuErrchk(cudaMalloc((void **)&d_symmat, sizeof(double) * n * n));
  gpuErrchk(cudaMalloc((void **)&d_mean, sizeof(double) * n));
  gpuErrchk(cudaMalloc((void **)&d_data, sizeof(double) * n * n));
  
  clock_gettime(CLOCK_REALTIME, rt + 0);
  /* Initialize array(s). */
	init_array(m, n, &float_n, data);  
  
  gpuErrchk(cudaMemcpy(d_symmat, symmat, sizeof(double) * n * n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_mean, mean, sizeof(double) * n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_data, data, sizeof(double) * n * n, cudaMemcpyHostToDevice));
  
	/* Run kernel. */
 
  //Calculate Block size
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE); 
 
	kernel_covariance<<<dimGrid, dimBlock>>>(m, n, float_n, d_data, d_symmat, d_mean);
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_REALTIME, rt + 1);
	wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("Covariance (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));   
  
  gpuErrchk(cudaMemcpy(symmat, d_symmat, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(data, d_data, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(mean, d_mean, sizeof(double) * n, cudaMemcpyDeviceToHost));
  
  int counter = 0;
  for (int i = 0; i<m*n; i++){
    //if(data[i] != data_correct[i]) counter += 1;
    //printf("%.2f\n", symmat[i]);
    //printf("%.2f\n", symmat_correct[i]);
    }
    
  //printf("%d", m*n);  
    
	gpuErrchk(cudaFree(data));
  gpuErrchk(cudaFree(d_symmat));
  gpuErrchk(cudaFree(d_mean));
  gpuErrchk(cudaFree(d_data));  
 
    
	cudaDeviceReset();

	//kernel_covariance(m, n, float_n, data, symmat, correct, mean);

	return 0;
}
