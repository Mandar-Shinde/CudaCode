

// some refrence from http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/Lecture5.pdf
//http://cs.nyu.edu/courses/fall14/CSCI-GA.3033-004/lecture3.pdf
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

__global__  void GPUMult(int *A, int *B, int *C, int WIDTH)
{	
	int sol=0;
	int i;i = threadIdx.x;
	int j; j= threadIdx.y;
	for (int cnt = 0; cnt < WIDTH;cnt++)
	{
		if (i != cnt)
			break;
		// 1D to 2D calc
		int h = WIDTH* cnt+ threadIdx.x; 
		//  |
		//	|
		//	V
		//	| move in y (loop for X constant)
		int w = WIDTH * threadIdx.y + cnt; //->-->--> move in x (loop for Y constant)

		sol += A[h]*B[w];
		//printf("[%d]  [%d][%d]   [%d]*[%d] ==%d\n",cnt, h, w, A[h], B[w], sol);
	}
	printf(" <> %d, %d    \n", i,j );
	C[i] = sol;

}

void CPUMult(int *array1, int *array2, int *result, int WIDTH)
{
}

int* prepareSquareMatrix(int isz, MATRIX_TYPE typ)
{
	int matSize = isz*isz;
	int *mat = (int*)malloc(matSize* sizeof(int));

	switch (typ)
	{
	case MATRIX_INITIALIZE:
		memset(mat, 0, matSize * sizeof(int));
		break;
	case MATRIX_RANDOM:
		printf(" ===============\n");
		for (long j = 0; j < (matSize); j++)
		{
			mat[j] = rand() % 10;
			printf(" == %d \n", mat[j]);
		}
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
	int *A, *B, *SOL;				// HOST
	int *cudaA, *cudaB, *cudaSUM, *cudaRET;	// DEVICE
	int msize = 3;  // 3X3

	// Initializing matrix with data
	A = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	B = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	SOL = prepareSquareMatrix(msize, MATRIX_INITIALIZE);  
	cudaRET = (int *)malloc(msize*msize*sizeof(int));
	memset(cudaRET, 0, msize*msize*sizeof(int));

	CPUMult(A, B, SOL, msize*msize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaA, msize*msize*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaB, msize*msize*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaSUM, msize*msize*sizeof(int)));

	// we are calculating 3X3 matrix so we will need only 9 threads
	int threadsPerBlock = 9;
	int blocksPerGrid = (msize*msize + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_SAFE_CALL(cudaMemcpy(cudaA, A, msize*msize*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cudaB, B, msize*msize*sizeof(int), cudaMemcpyHostToDevice));
	dim3 dimGrid(3, 3);
	dim3 dimBlock(3, 3);
	//GPUMult << <blocksPerGrid, threadsPerBlock >> >(cudaA, cudaB, cudaSUM, msize);
	GPUMult << <1, dimBlock >> >(cudaA, cudaB, cudaSUM, msize);
	_sleep(10);
	CUDA_SAFE_CALL(cudaMemcpy(cudaRET, cudaSUM, msize*msize*sizeof(int), cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < 9; i++)
		printf("\n [%d] =[%d]", i, cudaRET[i]);*/
	printf("\nBlocks per Grid :%d\nThreads pre Block :%d\n", blocksPerGrid, threadsPerBlock);
	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaSUM);
	free(A);
	free(B);
	free(SOL);
	cudaDeviceReset();

	return 0;
}
