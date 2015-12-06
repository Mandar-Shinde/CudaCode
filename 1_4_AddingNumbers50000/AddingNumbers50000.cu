/**
* Example from CUDA Sample 7
*/

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h> 
#include <stdio.h>
#include <conio.h>

/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numElements.
*/
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

void CUDA_SAFE_CALL(cudaError_t call)
{
	cudaError_t ret = call;
	if (ret != cudaSuccess)
	{
		printf(" ERROR at line :%i.%d' ' %s\n", __LINE__, ret, cudaGetErrorString(ret));
		exit(-1);
	}
}
/**
* Host main routine
*/
int main(void)
{
	cudaError_t err = cudaSuccess;
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);
	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// Allocate momory
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, size));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, size));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_C, size));

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy host data to device\n");
	CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("\nBlocks per Grid :%d\nThreads pre Block :%d\n", blocksPerGrid, threadsPerBlock);
	
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free device and host memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	// Reset the device and exit
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	_getche();
	return 0;
}

