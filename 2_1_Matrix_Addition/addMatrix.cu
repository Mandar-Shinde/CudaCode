/////////////////////////////////////////////////////
///  Code to add Matrix
///
/////////////////////////////////////////////////////
///  COMPILER OPTIONS
///
///
/////////////////////////////////////////////////////
///  OUTPUT
///
/////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory>

enum MATRIX_TYPE{ MATRIX_INITIALIZE, MATRIX_RANDOM, MATRIX_IDENTITY };

void CUDA_SAFE_CALL(cudaError_t call)
{
	cudaError_t ret = call;
	if (ret != cudaSuccess)
	{
		printf(" ERROR at line :%i.%d' ' %s\n", __LINE__, ret, cudaGetErrorString(ret));
		exit(-1);
	}
}

__global__  void GPUAdd(float *array1, float *array2, float *result, int WIDTH)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	result[threadIdx.x] = array1[threadIdx.x] + array2[threadIdx.x];
}

void CPUAdd(float *array1, float *array2, float *result, int WIDTH)
{
	for (int i = 0; i < WIDTH; i++)
		result[i] = array1[i] + array2[i];
}

float* prepareSquareMatrix(int isz, MATRIX_TYPE typ)
{
	int matSize = isz*isz;
	float *mat = (float*)malloc(matSize* sizeof(float));

	switch (typ)
	{
	case MATRIX_INITIALIZE:
		memset(mat, 0, matSize * sizeof(float));
		break;
	case MATRIX_RANDOM:
		for (long j = 0; j<(matSize); j++)
			mat[j] = (float)rand() / (float)RAND_MAX;
		break;
	case MATRIX_IDENTITY:
		// NOT IN USE
		break;
	}
	return mat;
}



int main()
{
	// Pointer for matrix
	float *A, *B, *SUM;				// HOST
	float *cudaA, *cudaB, *cudaSUM, *cudaRET;	// DEVICE
	int msize = 1000;
	
	// Initializing matrix with data
	A = prepareSquareMatrix(msize, MATRIX_RANDOM);  // 4 X 4
	B = prepareSquareMatrix(msize, MATRIX_RANDOM);  // 4 X 4
	SUM = prepareSquareMatrix(msize, MATRIX_INITIALIZE);  // 4 X 4
	cudaRET = (float *)malloc(msize*msize*sizeof(float));

	CPUAdd(A, B, SUM, msize*msize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaA, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaB, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaSUM, msize*msize*sizeof(float)));


	int threadsPerBlock = 256;
	int blocksPerGrid = (msize + threadsPerBlock - 1) / threadsPerBlock;
	printf("\nBlocks per Grid :%d\nThreads pre Block :%d\n", blocksPerGrid, threadsPerBlock);

	CUDA_SAFE_CALL(cudaMemcpy(cudaA, A, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cudaB, B, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	GPUAdd << <blocksPerGrid, threadsPerBlock >> >(cudaA, cudaB, cudaSUM, msize*msize);
	CUDA_SAFE_CALL(cudaMemcpy(cudaRET, cudaSUM, msize*msize, cudaMemcpyDeviceToHost));
	
	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaSUM);
	free(A);
	free(B);
	free(SUM);

    return 0;
}
