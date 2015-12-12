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

__global__  void addMatrix(float *array1, float *array2, float *result, int WIDTH)
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
		for (int j = 0; j<(matSize); j++)
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
	float *A, *B, *SUM;
	
	// Initializing matrix with data
	A = prepareSquareMatrix(4,MATRIX_RANDOM);  // 4 X 4
	B = prepareSquareMatrix(4, MATRIX_RANDOM);  // 4 X 4
	SUM = prepareSquareMatrix(4, MATRIX_INITIALIZE);  // 4 X 4



    return 0;
}
