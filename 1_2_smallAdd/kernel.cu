
#include "cuda_runtime.h"
#include <cuda.h> 
#include <stdio.h> 
#include <cuda.h>  

__global__   void add(int a, int b, int *c)
{

	*c = a + b;

}

/* Utility Macro : CUDA SAFE CALL */


void CUDA_SAFE_CALL(cudaError_t call)
{

	cudaError_t ret = call;

	switch (ret)
	{
	case cudaSuccess:

		break;

	default:
	{

		printf(" ERROR at line :%i.%d' ' %s\n",
			__LINE__, ret, cudaGetErrorString(ret));

		exit(-1);

		break;
	}
	}
}

int main(void)
{
	int c; int *dev_c;

	CUDA_SAFE_CALL(cudaMalloc((void**)& dev_c, sizeof(int)));

	add << < 1, 1 >> >(15, 13, dev_c);

	CUDA_SAFE_CALL(cudaMemcpy(
		&c,
		dev_c,
		sizeof(int),
		cudaMemcpyDeviceToHost));

	printf(" 15 + 13 = %d \n ", c);
	cudaFree(dev_c);
	return 0;
}
