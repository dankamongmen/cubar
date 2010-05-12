#include <stdlib.h>
#include "cuda8803ss.h"

__global__ void quirkykernel(void){
	int __shared__ sharedvar;

	sharedvar = 0;
	__syncthreads();
	while(sharedvar != threadIdx.x) /*** reconvergencepoint ***/ ;
	++sharedvar;
	__syncthreads();
}

int main(void){
	dim3 dg(1,1),db(64,1,1);
	CUresult cerr;

	if(init_cuda(0,NULL)){
		return EXIT_FAILURE;
	}
	quirkykernel<<<dg,db>>>();
	if((cerr = cuCtxSynchronize()) != CUDA_SUCCESS){
		fprintf(stderr,"Error running kernel (%d)\n",cerr);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
