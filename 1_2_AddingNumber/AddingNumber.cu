/////////////////////////////////////////////////////
///  Code to add number
///
/////////////////////////////////////////////////////
///  COMPILER OPTIONS
///
///  C:\Users\mandar\Documents\GitHub\CudaCode\1_2_AddingNumber>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" - gencode = arch = compute_20, code = \"sm_20,compute_20\" --use-local-env --cl-version 2013 -ccbin "C:\Program Files(x86)\Microsoft Visual Studio 12.0\VC\bin"  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include"  -G   --keep-dir Debug -maxrregcount=0  --machine 32 --compile -cudart static  -g   -DWIN32 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler " / EHsc / W3 / nologo / Od / Zi / RTC1 / MDd  " -o Debug\AddingNumber.cu.obj "C:\Users\mandar\Documents\GitHub\CudaCode\1_2_AddingNumber\AddingNumber.cu" 
///
/////////////////////////////////////////////////////
///  OUTPUT
///
///  15 + 13 = 28
/////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include <cuda.h> 
#include <stdio.h> 
#include <cuda.h>  

//
// This is the kernel which will execute on core
//
__global__   void add(int a, int b, int *c)
{
	*c = a + b;
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

int main(void)
{
	int c; int *dev_c;

	CUDA_SAFE_CALL(cudaMalloc((void**)& dev_c, sizeof(int)));

	add << < 1, 1 >> >(15, 13, dev_c);

	CUDA_SAFE_CALL(cudaMemcpy(&c,dev_c,sizeof(int),	cudaMemcpyDeviceToHost));

	printf(" 15 + 13 = %d \n ", c);
	cudaFree(dev_c);
	return 0;
}
