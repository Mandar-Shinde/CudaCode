/////////////////////////////////////////////////////
///  Code to Multiply Matrix
///
/////////////////////////////////////////////////////
///  COMPILER OPTIONS
/// 
//  C:\Users\mandar\Documents\GitHub\CudaCode\2_2_Matrix_Multiplication>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" - gencode = arch = compute_20, code = \"sm_20,compute_20\" --use-local-env --cl-version 2013 -ccbin "C:\Program Files(x86)\Microsoft Visual Studio 12.0\VC\bin"  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include"  -G   --keep-dir Debug -maxrregcount=0  --machine 32 --compile -cudart static  -g   -DWIN32 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler " / EHsc / W3 / nologo / Od / Zi / RTC1 / MDd  " -o Debug\multMatrix.cu.obj "C:\Users\mandar\Documents\GitHub\CudaCode\2_2_Matrix_Multiplication\multMatrix.cu" 
/////////////////////////////////////////////////////
///  OUTPUT
///
/*
// 5 X 5
150|  160|  170|  180|  190|
400|  435|  470|  505|  540|
650|  710|  770|  830|  890|
900|  985|  1070|  1155|  1240|
1150|  1260|  1370|  1480|  1590|

*/
// Some refrence from 
//http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/Lecture5.pdf
//http://cs.nyu.edu/courses/fall14/CSCI-GA.3033-004/lecture3.pdf
//http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
/////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory>
#include <math.h>

enum MATRIX_TYPE{ MATRIX_INITIALIZE, MATRIX_RANDOM, MATRIX_IDENTITY, MATRIX_ORDERED };

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

	if (i < WIDTH && j < WIDTH) {
		for (int k = 0; k < WIDTH; k++)
		{
			sol += A[j * WIDTH + k] * B[k * WIDTH + i];
		}
		C[j * WIDTH + i] = sol;	
	}

}

void CPUMult(int *A, int *B, int *C, int WIDTH)
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
		for (long j = 0; j < (matSize); j++)
			mat[j] =  rand() % 10;					
		break;
	case MATRIX_IDENTITY:
		// NOT IN USE
		break;
	case MATRIX_ORDERED:
		for (long j = 0; j < (matSize); j++)
		{
			mat[j] = j;
			//printf(" [%d] ", j);
		}
		break;
	}
	return mat;
}



int main()
{
	// Pointer for matrix
	int *A, *B, *SOL;				// HOST
	int *cudaA, *cudaB, *cudaMUL, *cudaRET;	// DEVICE
	int msize = 5;  

	// Initializing matrix with data
	A = prepareSquareMatrix(msize, MATRIX_ORDERED);
	B = prepareSquareMatrix(msize, MATRIX_ORDERED);
	SOL = prepareSquareMatrix(msize, MATRIX_INITIALIZE);  
	cudaRET = (int *)malloc(msize*msize*sizeof(int));
	memset(cudaRET, 0, msize*msize*sizeof(int));

	//CPUMult(A, B, SOL, msize*msize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaA, msize*msize*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaB, msize*msize*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaMUL, msize*msize*sizeof(int)));
	
	dim3 dimGrid(1, 1);
	dim3 dimBlock(msize, msize);

	CUDA_SAFE_CALL(cudaMemcpy(cudaA, A, msize*msize*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cudaB, B, msize*msize*sizeof(int), cudaMemcpyHostToDevice));

	GPUMult << <1, dimBlock >> >(cudaA, cudaB, cudaMUL, msize);

	CUDA_SAFE_CALL(cudaMemcpy(cudaRET, cudaMUL, msize*msize*sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0,k=0; i < msize; i++)
	{
		for (int j = 0; j < msize; j++)
		{
			printf(" %d| ", cudaRET[k]);
			k++;
		}
		printf(" \n");
	}

	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaMUL);
	free(A);
	free(B);
	free(SOL);
	cudaDeviceReset();

	return 0;
}
