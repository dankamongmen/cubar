#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
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
	size_t cudatmem,cudamem;
	if((cerr = cuMemGetInfo(&cudamem,&cudatmem)) != CUDA_SUCCESS){
		cuCtxDetach(ctx);
		goto err;
	}
	*mem = cudamem;
	*tmem = cudatmem;
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

#define GIDX ((gridDim.x * gridDim.y) * blockIdx.z + gridDim.x * blockIdx.y + \
		 blockIdx.x)

#define BIDX ((blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + \
		 threadIdx.x)

#define ABSIDX (((GIDX) * blockDim.x * blockDim.y * blockDim.z) + BIDX)

__global__ void memkernel(uint64_t *t0,uint64_t *t1){
	t0[ABSIDX] = clock64();
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void shlkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "shl.b32 %0, %1, %2;" : "=r"(pa) : "r"(pb), "r"(pc) );
		asm( "shl.b32 %0, %1, %2;" : "=r"(pa1) : "r"(pb1), "r"(pc1) );
		asm( "shl.b32 %0, %1, %2;" : "=r"(pb) : "r"(pc), "r"(pa) );
		asm( "shl.b32 %0, %1, %2;" : "=r"(pb1) : "r"(pc1), "r"(pa1) );
		asm( "shl.b32 %0, %1, %2;" : "=r"(pc) : "r"(pa), "r"(pb) );
		asm( "shl.b32 %0, %1, %2;" : "=r"(pc1) : "r"(pa1), "r"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void shrkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "shr.b32 %0, %1, %2;" : "=r"(pa) : "r"(pb), "r"(pc) );
		asm( "shr.b32 %0, %1, %2;" : "=r"(pa1) : "r"(pb1), "r"(pc1) );
		asm( "shr.b32 %0, %1, %2;" : "=r"(pb) : "r"(pc), "r"(pa) );
		asm( "shr.b32 %0, %1, %2;" : "=r"(pb1) : "r"(pc1), "r"(pa1) );
		asm( "shr.b32 %0, %1, %2;" : "=r"(pc) : "r"(pa), "r"(pb) );
		asm( "shr.b32 %0, %1, %2;" : "=r"(pc1) : "r"(pa1), "r"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void faddkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	double pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "add.f64 %0, %1, %2;" : "=d"(pa) : "d"(pb), "d"(pc) );
		asm( "add.f64 %0, %1, %2;" : "=d"(pa1) : "d"(pb1), "d"(pc1) );
		asm( "add.f64 %0, %1, %2;" : "=d"(pb) : "d"(pc), "d"(pa) );
		asm( "add.f64 %0, %1, %2;" : "=d"(pb1) : "d"(pc1), "d"(pa1) );
		asm( "add.f64 %0, %1, %2;" : "=d"(pc) : "d"(pa), "d"(pb) );
		asm( "add.f64 %0, %1, %2;" : "=d"(pc1) : "d"(pa1), "d"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void addkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "add.u32 %0, %1, %2;" : "=r"(pa) : "r"(pb), "r"(pc) );
		asm( "add.u32 %0, %1, %2;" : "=r"(pa1) : "r"(pb1), "r"(pc1) );
		asm( "add.u32 %0, %1, %2;" : "=r"(pb) : "r"(pc), "r"(pa) );
		asm( "add.u32 %0, %1, %2;" : "=r"(pb1) : "r"(pc1), "r"(pa1) );
		asm( "add.u32 %0, %1, %2;" : "=r"(pc) : "r"(pa), "r"(pb) );
		asm( "add.u32 %0, %1, %2;" : "=r"(pc1) : "r"(pa1), "r"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void add64kernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	uint64_t pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "add.u64 %0, %1, %2;" : "=l"(pa) : "l"(pb), "l"(pc) );
		asm( "add.u64 %0, %1, %2;" : "=l"(pa1) : "l"(pb1), "l"(pc1) );
		asm( "add.u64 %0, %1, %2;" : "=l"(pb) : "l"(pc), "l"(pa) );
		asm( "add.u64 %0, %1, %2;" : "=l"(pb1) : "l"(pc1), "l"(pa1) );
		asm( "add.u64 %0, %1, %2;" : "=l"(pc) : "l"(pa), "l"(pb) );
		asm( "add.u64 %0, %1, %2;" : "=l"(pc1) : "l"(pa1), "l"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void mulkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pa) : "r"(pb), "r"(pc) );
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pa1) : "r"(pb1), "r"(pc1) );
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pb) : "r"(pc), "r"(pa) );
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pb1) : "r"(pc1), "r"(pa1) );
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pc) : "r"(pa), "r"(pb) );
		asm( "mul.lo.u32 %0, %1, %2;" : "=r"(pc1) : "r"(pa1), "r"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void vaddr3kernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2;
	unsigned z;

	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pa) : "r"(pb), "r"(pc) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pa1) : "r"(pb1), "r"(pc1) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pb) : "r"(pc), "r"(pa) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pb1) : "r"(pc1), "r"(pa1) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pc) : "r"(pa), "r"(pb) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %2;" : "=r"(pc1) : "r"(pa1), "r"(pb1) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void vaddkernel(uint64_t *t0,uint64_t *t1,const unsigned loops){
	unsigned pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2,pd = 3,pd1 = 3;
	unsigned z;

	t0[ABSIDX] = clock64();
#pragma unroll 16
	for(z = 0 ; z < loops ; ++z){
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pa) : "r"(pb), "r"(pc), "r"(pd) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pa1) : "r"(pb1), "r"(pc1), "r"(pd1) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pb) : "r"(pc), "r"(pa), "r"(pd) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pb1) : "r"(pc1), "r"(pa1), "r"(pd1) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pc) : "r"(pa), "r"(pb), "r"(pd) );
		asm( "vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(pc1) : "r"(pa1), "r"(pb1), "r"(pd) );
	}
	t1[ABSIDX] = pc1 + pc;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

static void
stats(const struct timeval *tv0,const struct timeval *tv1,
		const uint64_t *t0,const uint64_t *t1,unsigned n,
		unsigned loops){
	uintmax_t sumdelt = 0;
	struct timeval tv;
	uint64_t res;
	unsigned z;

	res = *t1;
	timersub(tv1,tv0,&tv);
	printf("\tKernel wall time: %ld.%06lds\n",tv.tv_sec,tv.tv_usec);
	for(z = 0 ; z < n ; ++z){
		//printf("delt: %lu res: %u\n",t0[z],t1[z]);
		sumdelt += t0[z];
		assert(res == t1[z]);
	}
	printf("\tMean cycles / thread: %ju cycles / op: %ju\n",sumdelt / n,sumdelt / n / loops);
}

static int
check_const_ram(const unsigned loops){
	dim3 dblock(BLOCK_SIZE,1,1);
	struct timeval tv0, tv1;
	dim3 dgrid(1,1,1);
	uint64_t *h0,*h1;
	uint64_t *t0,*t1;
	size_t s;

	s = (dgrid.x * dgrid.y * dgrid.z) * (dblock.x * dblock.y * dblock.z);
	h0 = new uint64_t[s];
	h1 = new uint64_t[s];
	if(cudaMalloc(&t0,s * sizeof(*t0)) != cudaSuccess){
		fprintf(stderr,"\n  Error allocating %zu t0 bytes\n",s);
		free(h1); free(h0);
		return -1;
	}
	if(cudaMalloc(&t1,s * sizeof(*t0)) != cudaSuccess){
		fprintf(stderr,"\n  Error allocating %zu t0 bytes\n",s);
		cudaFree(t0); free(h1); free(h0);
		return -1;
	}

	printf("Timing 64-bit store+load+store...");
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	memkernel<<<dblock,dgrid>>>(t0,t1);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,1);

	printf("Timing %u adds...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	addkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u 64-bit adds...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	add64kernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u 64-bit floating-point adds...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	faddkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u muls...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	mulkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u vadds...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	vaddkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u vadds (duplicated registers)...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	vaddr3kernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u shls...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	shlkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	printf("Timing %u shrs...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	shrkernel<<<dblock,dgrid>>>(t0,t1,loops);
	if(cuCtxSynchronize() ||
			cudaMemcpy(h0,t0,s * sizeof(*h0),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(h1,t1,s * sizeof(*h1),cudaMemcpyDeviceToHost) != cudaSuccess){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error timing instruction (%s?)\n",
				cudaGetErrorString(err));
		goto err;
	}
	gettimeofday(&tv1,NULL);
	printf("good.\n");
	stats(&tv0,&tv1,h0,h1,s,loops * 6);

	cudaFree(t1); cudaFree(t0);
	free(h1); free(h0);
	return 0;

err:
	cudaFree(t1); cudaFree(t0);
	free(h1); free(h0);
	return -1;
}

#define LOOPS (0x00010000u)

static void
usage(const char *a0,int status){
	fprintf(stderr,"usage: %s [loops]\n",a0);
	fprintf(stderr," default loopcount: %u\n",LOOPS);
	exit(status);
}

int main(int argc,char **argv){
	unsigned long loops;
	int z,count;

	if(argc > 2){
		usage(argv[0],EXIT_FAILURE);
	}else if(argc == 2){
		if(getzul(argv[1],&loops)){
			usage(argv[0],EXIT_FAILURE);
		}
	}else{
		loops = LOOPS;
	}
	if(init_cuda_alldevs(&count)){
		return EXIT_FAILURE;
	}
	printf("CUDA device count: %d\n",count);
	for(z = 0 ; z < count ; ++z){
		uint64_t hostresarr[GRID_SIZE * BLOCK_SIZE];
		unsigned mem,tmem;
		uint64_t *resarr;
		int state;

		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem,&state)){
			return EXIT_FAILURE;
		}
		if(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) != cudaSuccess){
			fprintf(stderr,"Error preferring L1 to shared memory.\n");
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
		if(check_const_ram(loops)){
			return EXIT_FAILURE;
		}
		printf(" Success.\n");
		if(cudaMemcpy(hostresarr,resarr,sizeof(hostresarr),cudaMemcpyDeviceToHost) || cudaFree(resarr)){
			fprintf(stderr," Couldn't free result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
