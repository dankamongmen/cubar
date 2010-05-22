#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "cubar.h"

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,unsigned *mem,unsigned *tmem,int *state){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	void *str = NULL;
	CUcontext ctx;
	CUdevice c;

	*state = 0;
	if((cerr = cuDeviceGet(&c,dev)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't associative with device (%d)\n",cerr);
		return -1;
	}
	if((cerr = cudaGetDeviceProperties(&dprop,dev)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't get device properties (%d)\n",cerr);
		return -1;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,c);
	if(cerr != CUDA_SUCCESS || attr <= 0){
		return -1;
	}
	if((cerr = cuDeviceComputeCapability(&major,&minor,c)) != CUDA_SUCCESS){
		return -1;
	}
	if((str = malloc(CUDASTRLEN)) == NULL){
		return -1;
	}
	if((cerr = cuDeviceGetName((char *)str,CUDASTRLEN,c)) != CUDA_SUCCESS){
		goto err;
	}
	if((cerr = cuCtxCreate(&ctx,CU_CTX_MAP_HOST|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't create context (%d)\n",cerr);
		goto err;
	}
	if((cerr = cuMemGetInfo(mem,tmem)) != CUDA_SUCCESS){
		cuCtxDetach(ctx);
		goto err;
	}
	*state = dprop.computeMode;
	if(printf("%d.%d %s %s %u/%uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		*mem / (1024 * 1024) + !!(*mem / (1024 * 1024)),
		*tmem / (1024 * 1024) + !!(*tmem / (1024 * 1024)),
		*state == CU_COMPUTEMODE_EXCLUSIVE ? "(exclusive)" :
		*state == CU_COMPUTEMODE_PROHIBITED ? "(prohibited)" :
		*state == CU_COMPUTEMODE_DEFAULT ? "(shared)" :
		"(unknown compute mode)") < 0){
		cerr = -1;
		goto err;
	}
	free(str);
	return CUDA_SUCCESS;

err:	// cerr ought already be set!
	free(str);
	return cerr;
}

__device__ __constant__ unsigned constptr[1];

__global__ void constkernel(const unsigned *constmax){
	__shared__ unsigned psum[BLOCK_SIZE];
	unsigned *ptr;

	psum[threadIdx.x] = 0;
	// Accesses below 64k result in immediate termination, due to use of
	// the .global state space (2.0 provides unified addressing, which can
	// overcome this). That area's reserved for constant memory (.const
	// state space; see 5.1.3 of the PTX 2.0 Reference), from what I see.
	for(ptr = constptr ; ptr < constmax ; ptr += BLOCK_SIZE){
		++psum[threadIdx.x];
		if(ptr[threadIdx.x]){
			++psum[threadIdx.x];
		}
	}
	ptr = constptr;
	while((uintptr_t)ptr > threadIdx.x * sizeof(unsigned)){
		++psum[threadIdx.x];
		if(*(ptr - threadIdx.x)){
			++psum[threadIdx.x];
		}
		ptr -= BLOCK_SIZE;
	}
}

#define CONSTWIN ((unsigned *)0x10000u)
#define NOMANSPACE (0x1000u)

static int
check_const_ram(const unsigned *max){
	dim3 dblock(BLOCK_SIZE,1,1);
	dim3 dgrid(1,1,1);

	printf("  Verifying %jub constant memory...",(uintmax_t)max);
	fflush(stdout);
	constkernel<<<dblock,dgrid>>>(max);
	if(cuCtxSynchronize()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error verifying constant CUDA memory (%s?)\n",
				cudaGetErrorString(err));
		return -1;
	}
	printf("good.\n");
	return 0;
}

#define RANGER "out/cudaranger"

static int
divide_address_space(int devno,uintmax_t off,uintmax_t s,unsigned unit,
					unsigned gran,uint32_t *results,
					uintmax_t *worked){
	char min[40],max[40],dev[20];
	char * const argv[] = { RANGER, dev, min, max, NULL };
	pid_t pid;

	if((size_t)snprintf(dev,sizeof(dev),"%d",devno) >= sizeof(dev)){
		fprintf(stderr,"  Invalid device argument: %d\n",devno);
		return -1;
	}
	while(s){
		uintmax_t ts;
		int status;
		pid_t w;

		ts = s > gran ? gran : s;
		s -= ts;
		if((size_t)snprintf(min,sizeof(min),"0x%jx",off) >= sizeof(min) ||
			(size_t)snprintf(max,sizeof(max),"0x%jx",off + ts) >= sizeof(max)){
			fprintf(stderr,"  Invalid arguments: 0x%jx 0x%jx\n",off,off + ts);
			return -1;
		}
		off += ts;
		//printf("CALL: %s %s %s\n",dev,min,max);
		if((pid = fork()) < 0){
			fprintf(stderr,"  Couldn't fork (%s?)!\n",strerror(errno));
			return -1;
		}else if(pid == 0){
			if(execvp(RANGER,argv)){
				fprintf(stderr,"  Couldn't exec %s (%s?)!\n",RANGER,strerror(errno));
			}
			exit(CUDARANGER_EXIT_ERROR);
		}
		while((w = wait(&status)) != pid){
			if(w < 0){
				fprintf(stderr,"  Error waiting (%s?)!\n",
						strerror(errno));
				return -1;
			}
		}
		if(!WIFEXITED(status) || WEXITSTATUS(status) == CUDARANGER_EXIT_ERROR){
			fprintf(stderr,"  Exception running %s %s %s %s\n",
					argv[0],argv[1],argv[2],argv[3]);
			return -1;
		}else if(WEXITSTATUS(status) == CUDARANGER_EXIT_SUCCESS){
			*worked += ts;
		}else if(WEXITSTATUS(status) != CUDARANGER_EXIT_CUDAFAIL){
			fprintf(stderr,"  Unknown result code %d running"
				" %s %s %s %s\n",WEXITSTATUS(status),
				argv[0],argv[1],argv[2],argv[3]);
			return -1;
		} // otherwise, normal failure
	}
	return 0;
}

static int
cudadump(int devno,uintmax_t tmem,unsigned unit,uintmax_t gran,uint32_t *results){
	uintmax_t worked = 0,s;
	CUdeviceptr ptr;

	if(check_const_ram(CONSTWIN)){
		return -1;
	}
	if((s = cuda_alloc_max(stdout,&ptr,unit)) == 0){
		return -1;
	}
	printf("  Allocated %ju of %ju MB (%f%%) at 0x%jx:0x%jx\n",
			s / (1024 * 1024) + !!(s % (1024 * 1024)),
			tmem / (1024 * 1024) + !!(tmem % (1024 * 1024)),
			(float)s / tmem * 100,(uintmax_t)ptr,(uintmax_t)ptr + s);
	printf("  Verifying allocated region...\n");
	if(dump_cuda(ptr,ptr + (s / gran) * gran,unit,results)){
		cuMemFree(ptr);
		fprintf(stderr,"  Sanity check failed!\n");
		return -1;
	}
	if(cuMemFree(ptr)){
		fprintf(stderr,"  Error freeing CUDA memory (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
		return -1;
	}
	printf("  Dumping %jub...\n",tmem);
	if(divide_address_space(devno,NOMANSPACE,tmem,unit,gran,results,&worked)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	printf("  Readable: %jub/%jub (%f%%)\n",worked,tmem,(float)worked / tmem * 100);
	worked = 0;
	printf("  Dumping address space (%jub)...\n",(uintmax_t)0x100000000ull - NOMANSPACE);
	if(divide_address_space(devno,NOMANSPACE,0x100000000ull - NOMANSPACE,unit,gran,results,&worked)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	printf("  Readable: %jub/%jub (%f%%)\n",worked,0x100000000ull,(float)worked / 0x100000000 * 100);
	printf(" Success.\n");
	return 0;
}

#define GRAN_DEFAULT 4ul * 1024ul * 1024ul

static void
usage(const char *a0,int status){
	fprintf(stderr,"usage: %s [granularity]\n",a0);
	fprintf(stderr," default granularity: %lu\n",GRAN_DEFAULT);
	exit(status);
}

int main(int argc,char **argv){
	unsigned long gran;
	unsigned unit = 4;		// Minimum alignment of references
	int z,count;

	if(argc > 2){
		usage(argv[0],EXIT_FAILURE);
	}else if(argc == 2){
		if(getzul(argv[1],&gran)){
			usage(argv[0],EXIT_FAILURE);
		}
	}else{
		gran = GRAN_DEFAULT;
	}
	if(init_cuda_alldevs(&count)){
		return EXIT_FAILURE;
	}
	printf("CUDA device count: %d\n",count);
	for(z = 0 ; z < count ; ++z){
		uint32_t hostresarr[GRID_SIZE * BLOCK_SIZE];
		unsigned mem,tmem;
		uint32_t *resarr;
		int state;

		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem,&state)){
			return EXIT_FAILURE;
		}
		if(state != CU_COMPUTEMODE_DEFAULT){
			printf("  Skipping device %d (put it in shared mode).\n",z);
			continue;
		}
		if(cudaMalloc(&resarr,sizeof(hostresarr)) || cudaMemset(resarr,0,sizeof(hostresarr))){
			fprintf(stderr," Couldn't allocate result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
		if(cudadump(z,tmem,unit,gran,resarr)){
			return EXIT_FAILURE;
		}
		if(cudaMemcpy(hostresarr,resarr,sizeof(hostresarr),cudaMemcpyDeviceToHost) || cudaFree(resarr)){
			fprintf(stderr," Couldn't free result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
