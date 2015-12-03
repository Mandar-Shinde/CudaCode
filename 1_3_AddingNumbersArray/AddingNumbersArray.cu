
/////////////////////////////////////////////////////
///  Code to listdown attached compute device
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
#include <conio.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
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

int main()
{
	printf("Adding Using Cuda \n ");
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	CUDA_SAFE_CALL(addWithCuda(c, a, b, arraySize));

	printf("\n\n    [1,2,3,4,5]  +  [10,20,30,40,50] \n   = [%d, %d, %d, %d, %d]\n",c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CUDA_SAFE_CALL(cudaDeviceReset());
	
	_getche();
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus=cudaSuccess;

	// Choose which GPU to run on, change this on a multi-GPU system.
	CUDA_SAFE_CALL(cudaSetDevice(0));
	
	// Allocate GPU buffers for three vectors (two input, one output)    .
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_b, size * sizeof(int)));

	// Copy input vectors from host memory to GPU buffers.
	CUDA_SAFE_CALL(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));	
	CUDA_SAFE_CALL(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
	
	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	CUDA_SAFE_CALL(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CUDA_SAFE_CALL(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}