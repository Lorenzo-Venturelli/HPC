#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#define MINI_DATASET
#include "covariance.h"




/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
	int i, j;

	*float_n = 1.2;

	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
	 Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_2D(correct,M,M,m,m))

{
	int i, j;
	int ok = 0;

	for (i = 0; i<m; i++)
		for (j = 0; j < m; j++) {
			fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
			if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
			if (correct[i][j] != symmat[i][j]) {ok++;}
		}
	fprintf (stderr, "\n");
	printf("Corretto: %d", ok); 
}


/* Main computational kernel. The whole function will be timed,
	 including the call and return. */
static
void kernel_covariance(int m, int n,
					 DATA_TYPE float_n,
					 DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
					 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
					 DATA_TYPE POLYBENCH_2D(correct,M,M,m,m),
					 DATA_TYPE POLYBENCH_1D(mean,M,m))
{
	int i, j, j1, j2, k;
	
	/* Determine mean of column vectors of input data matrix */

	for (j = 0; j < _PB_M; j++)
	{
		mean[j] = 0.0;

		for (i = 0; i < _PB_N; i++)
			mean[j] += data[i][j];

		mean[j] /= float_n;

	/*for (k = _PB_N-1; k >=0; k--)
		data[j][k] -= mean[j];*/
	}
			
	/* Center the column vectors. */
	for (i = 0; i < _PB_N; i++)    
		for (j = 0; j < _PB_M; j++)
			data[i][j] -= mean[j];

			
	/* Calculate the m * m covariance matrix. */  
	#pragma omp parallel private(j1, j2, i) num_threads(4)
	{ 
		#pragma omp for schedule(dynamic, 16)
		for (j1 = 0; j1 < _PB_M; j1++)
		{
			for (j2 = j1; j2 < _PB_M; j2++)
			{
				symmat[j1][j2] = 0.0;
				for (i = 0; i < _PB_N; i++)
				{
					symmat[j1][j2] += data[i][j1] * data[i][j2];
				}
				symmat[j2][j1] = symmat[j1][j2];
			}
		}
	}


/*for (j1 = 0; j1 < _PB_M; j1++){
		
			for (j2 = j1; j2 < _PB_M; j2++)
	{
					correct[j1][j2] = 0.0;
		for (i = 0; i < _PB_N; i++)
			correct[j1][j2] += data[i][j1] * data[i][j2];
		correct[j2][j1] = correct[j1][j2];
				}
}*/
}

int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int m = M;
	
	//printf("n= %d", n);
	//printf("m= %d", m);

	/* Variable declaration/allocation. */
	DATA_TYPE float_n;
	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
	POLYBENCH_2D_ARRAY_DECL(correct,DATA_TYPE,M,M,m,m);
	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
	
	/* Initialize array(s). */
	init_array (m, n, &float_n, POLYBENCH_ARRAY(data));
	
	/* Start timer. */
	polybench_start_instruments;

	/* Run kernel. */
	kernel_covariance (m, n, float_n,
				 POLYBENCH_ARRAY(data),
				 POLYBENCH_ARRAY(symmat),
							POLYBENCH_ARRAY(correct),
				 POLYBENCH_ARRAY(mean));

	/* Stop and print timer. */
	polybench_stop_instruments;
	polybench_print_instruments;

	/* Prevent dead-code elimination. All live-out data must be printed
		 by the function call in argument. */
	polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);

	return 0;
}