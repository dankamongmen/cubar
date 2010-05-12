#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "cuda8803ss.h"

static int
dumpresults(const uint32_t *res,unsigned count){
	unsigned z,y,nonzero;

	nonzero = 0;
	for(z = 0 ; z < count ; z += 8){
		for(y = 0 ; y < 8 ; ++y){
			if(printf("%9x ",res[z + y]) < 0){
				return -1;
			}
			if(res[z + y]){
				++nonzero;
			}
		}
		if(printf("\n") < 0){
			return -1;
		}
	}
	if(nonzero == 0){
		fprintf(stderr,"  All-zero results. Kernel probably didn't run.\n");
		return -1;
	}
	return 0;
}

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno addrmin addrmax\n",a0);
}

int main(int argc,char **argv){
	uint32_t hostres[GRID_SIZE * BLOCK_SIZE],*resarr;
	unsigned long long min,max;
	unsigned unit = 4;		// Minimum alignment of references
	unsigned long zul;
	CUdeviceptr ptr;
	cudadump_e res;
	CUcontext ctx;
	char *eptr;
	int cerr;

	if(argc != 4){
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((zul = strtoul(argv[1],&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == argv[1] || *eptr){
		fprintf(stderr,"Invalid device number: %s\n",argv[1]);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((min = strtoull(argv[2],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[2] || *eptr){
		fprintf(stderr,"Invalid minimum address: %s\n",argv[2]);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((max = strtoull(argv[3],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[3] || *eptr){
		fprintf(stderr,"Invalid maximum address: %s\n",argv[3]);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(max <= min){
		fprintf(stderr,"Invalid arguments: max (%ju) <= min (%ju)\n",
				max,min);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if((cerr = init_cuda_ctx(zul,&ctx)) != CUDA_SUCCESS){
		fprintf(stderr,"Error initializing CUDA device %lu (%d, %s?)\n",
				zul,cerr,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cudaMalloc(&resarr,sizeof(hostres)) || cudaMemset(resarr,0x00,sizeof(hostres))){
		fprintf(stderr,"Error allocating %zu on device %lu (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cuda_alloc_max(NULL,&ptr,sizeof(unsigned)) == 0){
		fprintf(stderr,"Error allocating max on device %lu (%s?)\n",
			zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if((res = dump_cuda(min,max,unit,resarr)) != CUDARANGER_EXIT_SUCCESS){
		return res;
	}
	if(cudaThreadSynchronize()){
		return res;
	}
	if(cuMemFree(ptr)){
		fprintf(stderr,"Warning: couldn't free memory\n");
	}
	if(cudaMemcpy(hostres,resarr,sizeof(hostres),cudaMemcpyDeviceToHost)){
		fprintf(stderr,"Error copying %zu from device %lu (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cudaFree(resarr)){
		fprintf(stderr,"Couldn't free %zu on device %lu (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(dumpresults(hostres,sizeof(hostres) / sizeof(*hostres))){
		return CUDARANGER_EXIT_ERROR;
	}
	return CUDARANGER_EXIT_SUCCESS;
}
