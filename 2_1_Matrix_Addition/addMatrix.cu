/////////////////////////////////////////////////////
///  Code to add Matrix
///
/////////////////////////////////////////////////////
///  COMPILER OPTIONS
///
/// C:\Users\mandar\Documents\GitHub\CudaCode\2_1_Matrix_Addition>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin"  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include"  -G   --keep-dir Debug -maxrregcount=0  --machine 32 --compile   -g   -DWIN32 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od /Zi /RTC1 /MDd  " -o Debug\addMatrix.cu.obj "C:\Users\mandar\Documents\GitHub\CudaCode\2_1_Matrix_Addition\addMatrix.cu" -clean   addMatrix.cu
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
	result[i] = array1[i] + array2[i];
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
	int msize = 1000; //element size not width
	
	// Initializing matrix with data
	A = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	B = prepareSquareMatrix(msize, MATRIX_RANDOM);  
	SUM = prepareSquareMatrix(msize, MATRIX_INITIALIZE);  
	cudaRET = (float *)malloc(msize*msize*sizeof(float));
	memset(cudaRET, 0, msize*msize*sizeof(float));

	CPUAdd(A, B, SUM, msize*msize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaA, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaB, msize*msize*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&cudaSUM, msize*msize*sizeof(float)));


	int threadsPerBlock = 32;
	int blocksPerGrid = (msize*msize + threadsPerBlock - 1) / threadsPerBlock;
	printf("\nBlocks per Grid :%d\nThreads pre Block :%d\n", blocksPerGrid, threadsPerBlock);

	CUDA_SAFE_CALL(cudaMemcpy(cudaA, A, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cudaB, B, msize*msize*sizeof(float), cudaMemcpyHostToDevice));
	GPUAdd << <blocksPerGrid, threadsPerBlock >> >(cudaA, cudaB, cudaSUM, msize*msize);
	CUDA_SAFE_CALL(cudaMemcpy(cudaRET, cudaSUM, msize*msize*sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaSUM);
	free(A);
	free(B);
	free(SUM);

    return 0;
}
