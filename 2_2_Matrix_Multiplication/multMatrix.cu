

// some refrence from http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/Lecture5.pdf

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
		printf(" ERROR  :%d' ' %s\n", ret, cudaGetErrorString(ret));
		exit(-1);
	}
}

__global__  void GPUMult(float *A, float *B, float *C, int WIDTH)
{	
	float sol=0;
	for (int cnt = 0; cnt < WIDTH;cnt++)
	{
		// 1D to 2D calc
		int h = WIDTH* cnt+ threadIdx.x; 
		//  |
		//	|
		//	V
		//	| move in y (loop for X constant)
		int w = WIDTH * threadIdx.y + cnt; //->-->--> move in x (loop for Y constant)

		sol += A[h]*B[w];
	}
	C[i] = sol;
}

void CPUMult(float *array1, float *array2, float *result, int WIDTH)
{
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
	float *A, *B, *SOL;				// HOST
	float *cudaA, *cudaB, *cudaSUM, *cudaRET;	// DEVICE
	int msize = 3;  // 3X3

	// Initializing matrix with data
	A = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	B = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	SOL = prepareSquareMatrix(msize, MATRIX_INITIALIZE);  
	cudaRET = (float *)malloc(msize*msize*sizeof(float));
	memset(cudaRET, 0, msize*msize*sizeof(float));

	CPUMult(A, B, SOL, msize*msize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaA, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaB, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaSUM, msize*msize*sizeof(float)));

	// we are calculating 3X3 matrix so we will need only 9 threads
	int threadsPerBlock = 9;
	int blocksPerGrid = (msize*msize + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_SAFE_CALL(cudaMemcpy(cudaA, A, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cudaB, B, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	GPUMult << <blocksPerGrid, threadsPerBlock >> >(cudaA, cudaB, cudaSUM, msize*msize);
	_sleep(10);
	CUDA_SAFE_CALL(cudaMemcpy(cudaRET, cudaSUM, msize*msize*sizeof(float), cudaMemcpyDeviceToHost));


	printf("\nBlocks per Grid :%d\nThreads pre Block :%d\n", blocksPerGrid, threadsPerBlock);
	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaSUM);
	free(A);
	free(B);
	free(SOL);

	return 0;
}
