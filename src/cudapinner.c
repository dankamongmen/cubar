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

static const unsigned flags =
	CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;

static unsigned oldptr;

int main(int argc,char **argv){
	uintmax_t total = 0,s;
	unsigned long zul,sig;
	CUcontext ctx;
	void *ptr;
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
	if((s = cuda_hostalloc_max(stdout,&ptr,sizeof(unsigned),flags)) == 0){
		fprintf(stderr,"Error allocating pinnedmem on device %lu\n",zul);
		exit(EXIT_FAILURE);
	}
	zul = 0;
	do{
		if(printf("  Allocation at %p (expected 0x%x)\n",ptr,oldptr) < 0){
			exit(EXIT_FAILURE);
		}
		total += s;
		if((uintptr_t)ptr != oldptr){
			if(printf("  Memory hole: 0x%x->%p (%pb)\n",
				oldptr,(char *)ptr - 1,(char *)ptr - oldptr) < 0){
				exit(EXIT_SUCCESS);
			}
		}
		oldptr = (uintptr_t)ptr + s;
		++zul;
	}while( (s = cuda_hostalloc_max(stdout,&ptr,sizeof(unsigned),flags)) );
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
