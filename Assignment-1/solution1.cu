#include <cuda_runtime.h>
#include "kernels.h"

__global__ void per_row_kernel(int m, int n, int *A, int *B, int *C)
{
	int row = (blockIdx.x * blockDim.x) + (threadIdx.x); // This is a unique thread ID formula for 1D grid with 1D blocks.
	if(row < m)
	{
			for(int col = 0; col < n; col++)
			{
					
					C[row*n + col] = A[row*n + col] + B[row*n + col];
			}
	}
}

__global__ void per_column_kernel(int m, int n, int *A, int *B, int *C)
{
	int col =  (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y; // This is a unique thread ID formula for 1D grid with 2D blocks.
	if(col < n)
	{
			for(int row = 0; row < m; row++)
			{
					
					C[row*n + col] = A[row*n + col] + B[row*n + col];
			}
	}
}
__global__ void per_element_kernel(int m, int n, int *A, int *B, int *C)
{
	int tid = (blockIdx.x * gridDim.y * blockDim.x* blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y) +(threadIdx.x * blockDim.y) + threadIdx.y;   // Unique thread ID computation for 2D grid with 2D blocks

	// Extract the row-col IDs corresponding to the unique thread ID
	int row = tid / n;
	int col = tid % n;

	if(row < m && col < n)
	{
		C[row*n + col] = A[row*n + col] + B[row*n + col];
	}
}
