#include <cuda.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>
#include "cubar.h"

static unsigned thrdone,threadsmaintain = 1;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int
init_thread(CUcontext *pctx,CUdevice dev,size_t s){
	unsigned mfree,mtot;
	CUdeviceptr ptr;
	CUresult cerr;

	if( (cerr = cuCtxCreate(pctx,0,dev)) ){
		fprintf(stderr," Error (%d) creating CUDA context\n",cerr);
		return -1;
	}
	if(s){
		if( (cerr = cuMemAlloc(&ptr,s)) ){
			fprintf(stderr," Error (%d) allocating %zub\n",cerr,s);
			return -1;
		}
		printf("Allocated %zu (0x%zx)b at %x\n",s,s,ptr);
	}
	if(cuMemGetInfo(&mfree,&mtot)){
		return -1;
	}
	printf("%u of %u bytes free\n",mfree,mtot);
	return 0;
}

typedef struct ctx {
	unsigned long s;
	CUdevice dev;
	unsigned threadno;
} ctx;

static void *
thread(void *unsafectx){
	ctx x = *(ctx *)unsafectx;
	CUcontext cu;

	if(init_thread(&cu,x.dev,x.s)){
		goto err;
	}
	pthread_mutex_lock(&lock);
	printf("Got context at %p\n",cu);
	thrdone = 1;
	pthread_cond_broadcast(&cond);
	while(threadsmaintain){
		pthread_cond_wait(&cond,&lock);
	}
	pthread_mutex_unlock(&lock);
	return NULL;

err:
	pthread_mutex_lock(&lock);
	thrdone = 1;
	threadsmaintain = 0;
	pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&lock);
	return NULL;
}

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno perthreadbytes\n",a0);
}

int main(int argc,char **argv){
	unsigned total = 0;
	unsigned long zul;
	ctx marsh;

	if(argc != 3){
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if(getzul(argv[1],&zul)){
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if(getzul(argv[2],&marsh.s)){
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if(init_cuda(zul,&marsh.dev)){
		exit(EXIT_FAILURE);
	}
	while( (marsh.threadno = ++total) ){
		pthread_t tid;
		int err;

		if( (err = pthread_create(&tid,NULL,thread,&marsh)) ){
			fprintf(stderr,"Couldn't create thread %d (%s?)\n",
					total,strerror(err));
			exit(EXIT_SUCCESS);
		}
		pthread_mutex_lock(&lock);
		while(!thrdone && threadsmaintain){
			pthread_cond_wait(&cond,&lock);
		}
		thrdone = 0;
		if(!threadsmaintain){
			pthread_mutex_unlock(&lock);
			fprintf(stderr,"Thread %d exited with an error.\n",total);
			break;
		}
		pthread_mutex_unlock(&lock);
		printf("Created thread %d\n",total);
	}	
	exit(EXIT_SUCCESS);
}
