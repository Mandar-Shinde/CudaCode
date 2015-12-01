
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/* Utility Macro : CUDA SAFE CALL */
void CUDA_SAFE_CALL(cudaError_t call)
{
	cudaError_t ret = call;

	switch (ret)
	{

	case cudaSuccess:

		break;

	default:
	{ printf(" ERROR at line :%i.%d' ' %s\n",
		__LINE__, ret, cudaGetErrorString(ret));

	exit(-1);

	break;

	}
	}
}

int main()
{
	int computCount;
	cudaDeviceProp pro;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&computCount));

	for (int i = 0; i < computCount; i++)
	{
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&pro,i));
		printf("-------------------------------------------------");
		printf("Device ID: %d\n", computCount);
		printf("Device Name: %s\n", pro.name);
		printf("Compute :  %d.%d\n",pro.major, pro.minor);
		printf("Clock :  %d\n", pro.clockRate);
	
	}

    return 0;
}
