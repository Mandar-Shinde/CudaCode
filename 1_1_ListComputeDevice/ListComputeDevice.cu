/////////////////////////////////////////////////////
///  Code to listdown attached compute device
///
/////////////////////////////////////////////////////
///  COMPILER OPTIONS
///
/// "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" - gencode = arch = compute_20, code = \"sm_20,compute_20\" --use-local-env --cl-version 2013 -ccbin "C:\Program Files(x86)\Microsoft Visual Studio 12.0\VC\bin"  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include"  -G   --keep-dir Debug -maxrregcount=0  --machine 32 --compile -cudart static  -g   -DWIN32 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler " / EHsc / W3 / nologo / Od / Zi / RTC1 / MDd  " -o Debug\ListComputeDevice.cu.obj "C:\Users\mandar\Documents\GitHub\CudaCode\1_HelloWorld\ListComputeDevice.cu" 
///
/////////////////////////////////////////////////////
///  OUTPUT
///
//------------------------------------------------ -
//DeviceID: 1
//Device : GeForce GT 610
//Compute : 2.1
//Clock : 1620 MHz
//Memory : 2048 MB
//------------------------------------------------ -
///
/////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>

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
	int computCount;
	cudaDeviceProp pro;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&computCount));

	for (int i = 0; i < computCount; i++)
	{
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&pro, i));
		printf("-------------------------------------------------\n");
		printf("DeviceID: %d\n", computCount);
		printf("Device  : %s\n", pro.name);
		printf("Compute : %d.%d\n", pro.major, pro.minor);
		printf("Clock   : %d MHz \n", (pro.clockRate / 1000));
		printf("Memory  : %d MB \n", (pro.totalGlobalMem) / 1024 / 1024);

		// As Per Wikipedia 
		// Number of ALU lanes for integer and floating-point arithmetic operations	
		int icuda=0;	
		switch (pro.major)
		{
		case 1: 
			printf("Arch    : Tesla \n"); 
			icuda = pro.multiProcessorCount * 8;
			break;
		case 2:
			printf("Arch    : Fermi \n");
			if (pro.minor == 1)
				icuda = pro.multiProcessorCount * 48;
			else
				icuda = pro.multiProcessorCount * 32;
			break;
		case 3:
			printf("Arch   : Kepler \n"); 
			icuda = pro.multiProcessorCount * 192;
			break;
		case 4:
			printf("Arch   : Unknown \n");
			icuda = pro.multiProcessorCount * 1;
			break;
		case 5:
			printf("Arch   : Maxwell \n");
			icuda = pro.multiProcessorCount * 128;
			break;
		}
		printf("CUDA    : %d\n", icuda);

		printf("-------------------------------------------------\n");
		_getch();
	}
	return 0;
}



