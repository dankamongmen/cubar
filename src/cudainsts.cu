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

__global__ void memkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1){
	t0[ABSIDX] = clock64();
	t0[ABSIDX] = clock64() - t0[ABSIDX];
	t1[ABSIDX] = 0xffu;
}

#define ILP6OP2(inst,ptxtype,type) \
	uint32_t pa,pb = 1,pa1,pb1 = 1; \
	uint32_t pa2,pb2 = 3; \
	uint32_t pa3,pb3 = 10; \
	uint32_t pa4,pb4 = 100; \
	uint32_t pa5,pb5 = 1010; \
	unsigned z; \
\
	t0[ABSIDX] = clock64(); \
	for(z = 0 ; z < loops ; ++z){ \
		asm( \
		    inst "." ptxtype " %0, %1\n\t" \
		    inst "." ptxtype " %2, %3\n\t" \
		    inst "." ptxtype " %4, %5\n\t" \
		    inst "." ptxtype " %6, %7\n\t" \
		    inst "." ptxtype " %8, %9\n\t" \
		    inst "." ptxtype " %10, %11\n\t" \
		    inst "." ptxtype " %1, %0\n\t" \
		    inst "." ptxtype " %3, %2\n\t" \
		    inst "." ptxtype " %5, %4\n\t" \
		    inst "." ptxtype " %7, %6\n\t" \
		    inst "." ptxtype " %9, %8\n\t" \
		    inst "." ptxtype " %11, %10\n\t" \
		    inst "." ptxtype " %0, %1\n\t" \
		    inst "." ptxtype " %2, %3\n\t" \
		    inst "." ptxtype " %4, %5\n\t" \
		    inst "." ptxtype " %6, %7\n\t" \
		    inst "." ptxtype " %8, %9\n\t" \
		    inst "." ptxtype " %10, %11" \
		    : "=r"(pa), "+r"(pb), \
		      "=r"(pa1), "+r"(pb1), \
		      "=r"(pa2), "+r"(pb2), \
		      "=r"(pa3), "+r"(pb3), \
		      "=r"(pa4), "+r"(pb4), \
		      "=r"(pa5), "+r"(pb5) \
		); \
	}; \
	t1[ABSIDX] = pb + pb1 + pb2 + pb3 + pb4 + pb5; \
	t0[ABSIDX] = clock64() - t0[ABSIDX]

//#pragma unroll 128
#define ILP6OP3(inst,type,constraint) \
	type pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2; \
	type pa2,pb2 = 3,pc2 = 4; \
	type pa3,pb3 = 10,pc3 = 20; \
	type pa4,pb4 = 100,pc4 = 201; \
	type pa5,pb5 = 1010,pc5 = 2001; \
	unsigned z; \
\
	t0[ABSIDX] = clock64(); \
	for(z = 0 ; z < loops ; ++z){ \
		asm( \
		inst " %0, %1, %2\n\t" \
		inst " %3, %4, %5\n\t" \
		inst " %6, %7, %8\n\t" \
		inst " %9, %10, %11\n\t" \
		inst " %12, %13, %14\n\t" \
		inst " %15, %16, %17\n\t" \
		inst " %1, %2, %0\n\t" \
		inst " %4, %5, %3\n\t" \
		inst " %7, %8, %6\n\t" \
		inst " %10, %11, %9\n\t" \
		inst " %13, %12, %14\n\t" \
		inst " %16, %15, %17\n\t" \
		inst " %2, %1, %0\n\t" \
		inst " %5, %4, %3\n\t" \
		inst " %8, %7, %6\n\t" \
		inst " %11, %10, %9\n\t" \
		inst " %14, %13, %12\n\t" \
		inst " %17, %16, %15" \
		: "=" constraint (pa), "+" constraint (pb), "+" constraint (pc) \
		  "=" constraint (pa1), "+" constraint (pb1), "+" constraint (pc1) \
		  "=" constraint (pa2), "+" constraint (pb2), "+" constraint (pc2) \
		  "=" constraint (pa3), "+" constraint (pb3), "+" constraint (pc3) \
		  "=" constraint (pa4), "+" constraint (pb4), "+" constraint (pc4) \
		  "=" constraint (pa5), "+" constraint (pb5), "+" constraint (pc5) \
		); \
	} \
	t1[ABSIDX] = pc + pc1 + pc2 + pc3 + pc4 + pc5; \
	t0[ABSIDX] = clock64() - t0[ABSIDX]

#define ILP6OP4(inst,ptxtype,type,constraint) \
	type pa,pb = 1,pc = 2,pa1,pb1 = 1,pc1 = 2,pd = 8, pd1 = 42; \
	type pa2,pb2 = 3,pc2 = 4, pd2 = 69; \
	type pa3,pb3 = 10,pc3 = 20, pd3 = 120; \
	type pa4,pb4 = 100,pc4 = 201, pd4 = 420; \
	type pa5,pb5 = 1010,pc5 = 2001, pd5 = 31337; \
	unsigned z; \
\
	t0[ABSIDX] = clock64(); \
	for(z = 0 ; z < loops ; ++z){ \
		asm( \
		inst "." ptxtype " %0, %1, %2, %18\n\t" \
		inst "." ptxtype " %3, %4, %5, %19\n\t" \
		inst "." ptxtype " %6, %7, %8, %20\n\t" \
		inst "." ptxtype " %9, %10, %11, %21\n\t" \
		inst "." ptxtype " %12, %13, %14, %22\n\t" \
		inst "." ptxtype " %15, %16, %17, %23\n\t" \
		inst "." ptxtype " %1, %2, %0, %18\n\t" \
		inst "." ptxtype " %4, %5, %3, %19\n\t" \
		inst "." ptxtype " %7, %8, %6, %20\n\t" \
		inst "." ptxtype " %10, %11, %9, %21\n\t" \
		inst "." ptxtype " %13, %12, %14, %22\n\t" \
		inst "." ptxtype " %16, %15, %17, %23\n\t" \
		inst "." ptxtype " %2, %1, %0, %18\n\t" \
		inst "." ptxtype " %5, %4, %3, %19\n\t" \
		inst "." ptxtype " %8, %7, %6, %20\n\t" \
		inst "." ptxtype " %11, %10, %9, %21\n\t" \
		inst "." ptxtype " %14, %13, %12, %22\n\t" \
		inst "." ptxtype " %17, %16, %15, %23" \
		: "=" constraint (pa), "+" constraint (pb), "+" constraint (pc), \
		  "=" constraint (pa1), "+" constraint (pb1), "+" constraint (pc1), \
		  "=" constraint (pa2), "+" constraint (pb2), "+" constraint (pc2), \
		  "=" constraint (pa3), "+" constraint (pb3), "+" constraint (pc3), \
		  "=" constraint (pa4), "+" constraint (pb4), "+" constraint (pc4), \
		  "=" constraint (pa5), "+" constraint (pb5), "+" constraint (pc5) \
		: constraint(pd), constraint(pd1), constraint(pd2), \
		  constraint(pd3), constraint(pd4), constraint(pd5) \
		); \
	} \
	t1[ABSIDX] = pc + pc1 + pc2 + pc3 + pc4 + pc5; \
	t0[ABSIDX] = clock64() - t0[ABSIDX]

__global__ void shrkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("shr.b32",uint32_t,"r");
}

__global__ void shlkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("shl.b32",uint32_t,"r");
}

__global__ void faddkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("add.f64",double,"d");
}

__global__ void popdepkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	unsigned pa = 0,pb = 1;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 128
	for(z = 0 ; z < loops ; ++z){
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "popc.b32 %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "popc.b32 %1, %0;" : "+r"(pb) : "r"(pa) );
	}
	t1[ABSIDX] = pa;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void shrdepkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	unsigned pa = 0,pb = 1;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 128
	for(z = 0 ; z < loops ; ++z){
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "shr.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
	}
	t1[ABSIDX] = pa;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void adddepkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	unsigned pa = 0,pb = 1;
	unsigned z;


	t0[ABSIDX] = clock64();
#pragma unroll 128
	for(z = 0 ; z < loops ; ++z){
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pa) : "r"(pb) );
		asm( "add.u32 %0, %1, %0;" : "+r"(pb) : "r"(pa) );
	}
	t1[ABSIDX] = pa;
	t0[ABSIDX] = clock64() - t0[ABSIDX];
}

__global__ void addkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("add.u32",uint32_t,"r");
}

__global__ void addcckernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("add.cc.u32",uint32_t,"r");
}

__global__ void add64kernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("add.u64",uint64_t,"l");
}

__global__ void mulkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP3("mul.lo.u32",uint32_t,"r");
}

__global__ void popkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP2("popc","b32",uint32_t);
}

__global__ void vaddkernel(uint64_t * __restrict__ t0,uint64_t * __restrict__ t1,const unsigned loops){
	ILP6OP4("vadd","u32.u32.u32.add",uint32_t,"r");
}

static void
stats(const struct timeval *tv0,const struct timeval *tv1,
		const uint64_t *t0,const uint64_t *t1,unsigned n,
		unsigned loops){
	uintmax_t sumdelt = 0,min = 0,max = 0;
	struct timeval tv;
	uint64_t res;
	unsigned z;

	res = *t1;
	timersub(tv1,tv0,&tv);
	printf("\tKernel wall time: %ld.%06lds Threads: %u\n",
			tv.tv_sec,tv.tv_usec,n);
	for(z = 0 ; z < n ; ++z){
		//printf("delt: %lu res: %u\n",t0[z],t1[z]);
		sumdelt += t0[z];
		if(t0[z] < min || min == 0){
			min = t0[z];
		}
		if(t0[z] > max){
			max = t0[z];
		}
		assert(res == t1[z]);
	}
	printf("\tMin cycles / thread: %ju cycles / op: %ju\n",
			min,min / loops);
	printf("\tAvg cycles / thread: %ju cycles / op: %ju\n",
			sumdelt / n,sumdelt / n / loops);
	printf("\tMax cycles / thread: %ju cycles / op: %ju\n",
			max,max / loops);
}

static int
test_inst_throughput(const unsigned loops){
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

	printf("Timing %u dependent adds...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	adddepkernel<<<dblock,dgrid>>>(t0,t1,loops);
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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

	printf("Timing %u add.ccs...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	addcckernel<<<dblock,dgrid>>>(t0,t1,loops);
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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

	printf("Timing %u pops...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	popkernel<<<dblock,dgrid>>>(t0,t1,loops);
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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

	printf("Timing %u dependent pops...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	popdepkernel<<<dblock,dgrid>>>(t0,t1,loops);
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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

	printf("Timing %u dependent shrs...",loops);
	fflush(stdout);
	gettimeofday(&tv0,NULL);
	shrdepkernel<<<dblock,dgrid>>>(t0,t1,loops);
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
	stats(&tv0,&tv1,h0,h1,s,loops * 18);

	cudaFree(t1); cudaFree(t0);
	delete[] h1; delete[] h0;
	return 0;

err:
	cudaFree(t1); cudaFree(t0);
	delete[] h1; delete[] h0;
	return -1;
}

#define LOOPS (0x00080000u)

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
		if(test_inst_throughput(loops)){
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
