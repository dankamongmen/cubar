#ifndef CUDA8803SS
#define CUDA8803SS

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

int init_cuda_alldevs(int *);
int init_cuda(int,CUdevice *);
int init_cuda_ctx(int,CUcontext *);
int getzul(const char *,unsigned long *);
uintmax_t cuda_alloc_max(FILE *,CUdeviceptr *,unsigned);
uintmax_t cuda_hostalloc_max(FILE *,void **,unsigned,unsigned);

int kernel_version(void);
int kernel_registry(void);
int kernel_version_str(void);
int kernel_cardinfo(unsigned);

#define CUDAMAJMIN(v) (v) / 1000, (v) % 1000

#ifdef __CUDACC__

	// this is some epic bullshit, done to work around issues in NVIDIA's
	// nvcc compiler...apologies all around

#define GRID_SIZE 1
#define BLOCK_SIZE 192

#include <sys/time.h>

// Result codes. _CUDAFAIL means that the CUDA kernel raised an exception -- an
// expected mode of failure. _ERROR means some other exception occurred (abort
// the binary search of the memory).
typedef enum {
	CUDARANGER_EXIT_SUCCESS,
	CUDARANGER_EXIT_ERROR,
	CUDARANGER_EXIT_CUDAFAIL,
} cudadump_e;

// Iterates over the specified memory region in units of CUDA's unsigned int.
// bptr must not be less than aptr. The provided array must be BLOCK_SIZE
// 32-bit integers; it holds the number of non-0 words seen by each of the
// BLOCK_SIZE threads.
__global__ void
readkernel(unsigned *aptr,const unsigned *bptr,uint32_t *results){
	__shared__ typeof(*results) psum[GRID_SIZE * BLOCK_SIZE];

	psum[blockDim.x * blockIdx.x + threadIdx.x] =
		results[blockDim.x * blockIdx.x + threadIdx.x];
	while(aptr + blockDim.x * blockIdx.x + threadIdx.x < bptr){
		++psum[blockDim.x * blockIdx.x + threadIdx.x];
		if(aptr[blockDim.x * blockIdx.x + threadIdx.x]){
			++psum[blockDim.x * blockIdx.x + threadIdx.x];
		}
		aptr += blockDim.x * gridDim.x;
	}
	results[blockDim.x * blockIdx.x + threadIdx.x] =
		psum[blockDim.x * blockIdx.x + threadIdx.x];
}

static inline cudadump_e
dump_cuda(uintmax_t tmin,uintmax_t tmax,unsigned unit,uint32_t *results){
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	int punit = 'M',cerr;
	dim3 dgrid(GRID_SIZE,1,1);
	uintmax_t usec,s;
	float bw;

	if(cudaThreadSynchronize()){
		fprintf(stderr,"   Error prior to running kernel (%s)\n",
				cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   readkernel {%ux%u} x {%ux%ux%u} (0x%08jx, 0x%08jx (0x%jxb), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	readkernel<<<dgrid,dblock>>>((unsigned *)tmin,(unsigned *)tmax,results);
	if((cerr = cudaThreadSynchronize()) != 0){ // FIXME workaround nvcc 4.0
		cudaError_t err;

		if(cerr != CUDA_ERROR_LAUNCH_FAILED && cerr != CUDA_ERROR_DEINITIALIZED){
			err = cudaGetLastError();
			fprintf(stderr,"   Error running kernel (%d, %s?)\n",
					cerr,cudaGetErrorString(err));
			return CUDARANGER_EXIT_ERROR;
		}
		//fprintf(stderr,"   Minor error running kernel (%d, %s?)\n",
				//cerr,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_CUDAFAIL;
	}
	gettimeofday(&time1,NULL);
	timersub(&time1,&time0,&timer);
	usec = (timer.tv_sec * 1000000 + timer.tv_usec);
	bw = (float)s / usec;
	if(bw > 1000.0f){
		bw /= 1000.0f;
		punit = 'G';
	}
	printf("   elapsed time: %ju.%jus (%.3f %cB/s) res: %d\n",
			usec / 1000000,usec % 1000000,bw,punit,cerr);
	return CUDARANGER_EXIT_SUCCESS;
}
#endif

#ifdef __cplusplus
};
#endif

#endif
