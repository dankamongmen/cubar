#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include "cubar.h"

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno [signal]\n",a0);
}

int main(int argc,char **argv){
	CUdeviceptr oldptr = 0,ptr;
	uintmax_t total = 0,s;
	unsigned long zul,sig;
	CUcontext ctx;
	int cerr;

	if(argc > 3 || argc < 2){
		usage(*argv);
		exit(EXIT_FAILURE);
	}else if(argc == 3){
		if(getzul(argv[2],&sig)){
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
		if(sig >= (unsigned)SIGRTMIN){
			fprintf(stderr,"Invalid signal (%lu > %d)\n",sig,SIGRTMIN);
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}else{
		sig = SIGRTMIN;
	}
	if(getzul(argv[1],&zul)){
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if((cerr = init_cuda_ctx(zul,&ctx)) != CUDA_SUCCESS){
		fprintf(stderr,"Error initializing CUDA device %lu (%d)\n",zul,cerr);
		exit(EXIT_FAILURE);
	}
	if((s = cuda_alloc_max(stdout,&ptr,sizeof(unsigned))) == 0){
		fprintf(stderr,"Error allocating max on device %lu\n",zul);
		exit(EXIT_FAILURE);
	}
	zul = 0;
	do{
		if(printf("  Allocation at 0x%llx (expected 0x%llx)\n",ptr,oldptr) < 0){
			exit(EXIT_FAILURE);
		}
		total += s;
		if(ptr != oldptr){
			if(printf("  Memory hole: 0x%llx->0x%llx (0x%llxb)\n",
				oldptr,ptr - 1,ptr - oldptr) < 0){
				exit(EXIT_SUCCESS);
			}
		}
		oldptr = ptr + s;
		++zul;
	}while( (s = cuda_alloc_max(stdout,&ptr,sizeof(unsigned))) );
	printf(" Got %ju (0x%jx) total bytes in %lu allocations.\n",total,total,zul);
	if(sig < (unsigned)SIGRTMIN){
		sigset_t set;
		int sigrx;

		printf("Waiting on signal %lu...\n",sig);
		if(sigemptyset(&set) || sigaddset(&set,sig) || sigwait(&set,&sigrx)){
			fprintf(stderr,"Error waiting on signal %lu (%s?)\n",
					sig,strerror(errno));
			exit(EXIT_FAILURE);
		}
		printf("Received signal %d, exiting.\n",sigrx);
	}
	exit(EXIT_SUCCESS);
}
