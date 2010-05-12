#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include "cuda8803ss.h"

static void
usage(const char *argv){
	fprintf(stderr,"usage: %s [ signo ]\n",argv);
}

int main(int argc,char **argv){
	unsigned long zul;
	CUresult cerr;
	int count;

	if(argc > 2){
		usage(*argv);
		exit(EXIT_FAILURE);
	}else if(argc == 2){
		if(getzul(argv[1],&zul)){
			usage(*argv);
			exit(EXIT_FAILURE);
		}
	}
	if( (cerr = cuInit(0)) ){
		fprintf(stderr,"Couldn't initialize CUDA (%d)\n",cerr);
		exit(EXIT_FAILURE);
	}
	printf("CUDA initialized.\n");
	if( (cerr = cuDeviceGetCount(&count)) ){
		fprintf(stderr,"Couldn't get CUDA dev count (%d)\n",cerr);
		exit(EXIT_FAILURE);
	}
	printf("%d devices\n",count);
	if(argc == 2){
		sigset_t s;
		int sig;

		if(sigemptyset(&s) || sigaddset(&s,zul) || sigwait(&s,&sig)){
			fprintf(stderr,"Error waiting for signal %lu (%s?)\n",
					zul,strerror(errno));
			exit(EXIT_FAILURE);
		}
	}
	exit(EXIT_SUCCESS);
}
