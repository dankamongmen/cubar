#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <sys/stat.h>
#include "cubar.h"

#define NVPROCDIR "/proc/driver/nvidia"
#define PROC_VERFILE "/proc/driver/nvidia/version"
#define PROC_REGISTRY "/proc/driver/nvidia/registry"

int init_cuda(int devno,CUdevice *c){
	int attr,cerr;
	CUdevice tmp;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize CUDA (%d), exiting.\n",cerr);
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if(c == NULL){
		c = &tmp; // won't be passed pack, but allows device binding
	}
	if((cerr = cuDeviceGet(c,devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get device reference (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

int init_cuda_ctx(int devno,CUcontext *cu){
	CUdevice c;
	int cerr;

	if((cerr = init_cuda(devno,&c)) != CUDA_SUCCESS){
		return cerr;
	}
	if((cerr = cuCtxCreate(cu,CU_CTX_SCHED_YIELD| CU_CTX_MAP_HOST,c)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't create context (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

uintmax_t cuda_hostalloc_max(FILE *o,void **ptr,unsigned unit,unsigned flags){
	uintmax_t tmax = 1ull << ADDRESS_BITS;
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cuMemHostAlloc(ptr,s,flags)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s < tmax){
			int cerr;

			if(o){ fprintf(o,"%jub @ %p...",s,*ptr); }
			if((cerr = cuMemFreeHost(*ptr)) ){
				fprintf(stderr,"  Couldn't free %jub at %p (%d?)\n",
					s,*ptr,cerr);
				return 0;
			}
			min = s;
		}else{
			if(o) { fprintf(o,"%jub!\n",s); }
			return s;
		}
	}while( (s = ((tmax + min) * unit / 2 / unit)) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

uintmax_t cuda_alloc_max(FILE *o,CUdeviceptr *ptr,unsigned unit){
	uintmax_t tmax = 1ull << ADDRESS_BITS;
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cuMemAlloc(ptr,s)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s < tmax){
			int cerr;

			if(o){ fprintf(o,"%jub @ 0x%llx...",s,*ptr); }
			printf("min/max: %ju %ju\n",min,tmax);
			if((cerr = cuMemFree(*ptr)) ){
				fprintf(stderr,"  Couldn't free %jub at 0x%llx (%d?)\n",
					s,*ptr,cerr);
				return 0;
			}
			if(min < s){
				min = s;
			}else{
				min += unit;
			}
		}else{
			if(o) { fprintf(o,"%jub!\n",s); }
			return s;
		}
	}while( (s = ((tmax + min) * unit / 2 / unit)) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

int getzul(const char *arg,unsigned long *zul){
	char *eptr;

	if(((*zul = strtoul(arg,&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == arg || *eptr){
		fprintf(stderr,"Expected an unsigned integer, got \"%s\"\n",arg);
		return -1;
	}
	return 0;
}

static int
dumpprocfile(const char *fn){
	char c[256];
	ssize_t r;
	int fd;

	if((fd = open(fn,O_RDONLY)) < 0){
		fprintf(stderr,"Couldn't open %s (%s)\n",fn,strerror(errno));
		return -1;
	}
	while((r = read(fd,c,sizeof(c))) > 0){
		if(printf("%.*s",(int)r,c) < r){
			close(fd);
			return -1;
		}
		if((size_t)r < sizeof(c)){
			break;
		}
	}
	if(r < 0){
		fprintf(stderr,"Error reading %s (%s)\n",fn,strerror(errno));
		close(fd);
		return -1;
	}
	if(fflush(stdout)){
		close(fd);
		return -1;
	}
	if(close(fd)){
		fprintf(stderr,"Error closing %s (%s)\n",fn,strerror(errno));
		return -1;
	}
	return 0;
}

// FIXME this ought also duplicate nvidia-smi -q behavior: temp, pciid, ECC, etc
int kernel_cardinfo(unsigned idx){
	char fn[NAME_MAX];
	int r;

	if((r = snprintf(fn,sizeof(fn),"%s/cards/%u",NVPROCDIR,idx)) < 0){
		return -1;
	}
	if((unsigned)r >= sizeof(fn)){
		return -1;
	}
	return dumpprocfile(fn);
}

int kernel_registry(void){
	return dumpprocfile(PROC_REGISTRY);
}

int kernel_version(void){
	return dumpprocfile(PROC_VERFILE);
}

int kernel_version_str(void){
#define VERFILEMAX ((size_t)1024)
#define NVRMTAG "NVRM version: "
	char *nvrmver,*tok;
	ssize_t b,nl;
	int fd;

	if((fd = open(PROC_VERFILE,O_RDONLY)) < 0){
		fprintf(stderr,"Couldn't open %s (%s)\n",PROC_VERFILE,strerror(errno));
		return -1;
	}
	if((nvrmver = (char *)malloc(VERFILEMAX)) == NULL){
		fprintf(stderr,"Couldn't allocate %zub readbuf (%s)\n",VERFILEMAX,strerror(errno));
		close(fd);
		return -1;
	}
	if((b = read(fd,nvrmver,VERFILEMAX)) < 0){
		fprintf(stderr,"Error reading %s (%s)\n",PROC_VERFILE,strerror(errno));
		free(nvrmver);
		close(fd);
		return -1;
	}
	for(nl = 0 ; nl < b ; ++nl){
		if(nvrmver[nl] == '\n'){
			nvrmver[nl] = '\0';
			break;
		}
	}
	if(nl == b || (tok = strstr(nvrmver,NVRMTAG)) == NULL){
		fprintf(stderr,"Badly-formed version file at %s\n",PROC_VERFILE);
		free(nvrmver);
		close(fd);
		return -1;
	}
	tok += strlen(NVRMTAG);
	printf("%s\n",tok);
	free(nvrmver);
	close(fd);
	return 0;
#undef NVRMTAG
#undef VERFILEMAX
}

int init_cuda_alldevs(int *count){
	int attr,cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize CUDA (%d)\n",cerr);
		return -1;
	}
	if(kernel_version_str()){
		return -1;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get CUDA driver version (%d)\n",cerr);
		return -1;
	}
	printf("Compiled against CUDA version %d.%d. Linked against CUDA version %d.%d.\n",
			CUDAMAJMIN(CUDA_VERSION),CUDAMAJMIN(attr));
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if((cerr = cuDeviceGetCount(count)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get CUDA device count (%d)\n",cerr);
		return -1;
	}
	if(*count <= 0){
		fprintf(stderr,"No CUDA devices found, exiting.\n");
		return -1;
	}
	return 0;
}
