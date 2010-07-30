#include <errno.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
// As of 3.1.7, at least, libpci doesn't use C++ guards :/
#ifdef __cplusplus
extern "C" {
#endif
#include <pci/pci.h>		// libpci
#ifdef __cplusplus
}
#endif
#include <sys/wait.h>
#include <sys/mman.h>
#include <readline/history.h>	// GNU readline
#include <readline/readline.h>	// GNU readline
#include "cubar.h"

#define HISTORY_FILE ".cudahistory" // FIXME use homedir
// Single bytes allocated using cuMemAlloc() have a periodicity of 0x100, but
// this might change in the future (or past).
#define MINIMUM_ALLOC 0x100

// FIXME can we not look this up at runtime in the PCI database, as well?
// http://www.pcidatabase.com/vendor_details.php?id=606
#define NVIDIA_VENDORID 0x10DE
#define PCI_CONFIG_BYTES 64
#define PCI_VGA_CLASS 0x03 // only the first byte of the (2-byte) device class

#define ENFORCE_ARGEND(c,cmdline)					\
	do{								\
		while(isspace(*cmdline)){ ++cmdline; }			\
		if(strcmp(cmdline,"")){					\
			if(fprintf(stderr,"too many options to %s\n",c) < 0){	\
				return -1;				\
			}						\
			return 0;					\
		}							\
	}while(0)

typedef struct cudamap {
	uintptr_t base;
	size_t s;		// only what we asked for, not actually got
	struct cudamap *next;
	void *maps;		// only for on-device mappings of host memory.
				// otherwise, equal to MAP_FAILED.
	unsigned allocno;	// 0 == cudash internal alloc
} cudamap;

typedef struct cudadev {
	char *devname;
	unsigned devno;
	struct cudadev *next;
	int major,minor,warpsz,mpcount,conkernels;
	int thrperblk;
	CUcontext ctx;
	cudamap *map;
	unsigned addrbits;
	CUdeviceptr resarray;
	unsigned alloccount;
} cudadev;

static cudamap *maps;
struct pci_access *pci;
static unsigned cudash_child;
static cudadev *devices,*curdev,systemdev;

static void
free_maps(cudamap *m){
	while(m){
		cudamap *tm = m;

		m = tm->next;
		free(tm);
	}
}

static void
free_devices(cudadev *d){
	while(d){
		cudadev *t = d;
		int cerr;

		d = d->next;
		free(t->devname);
		free_maps(t->map);
		if((cerr = cuMemFree(t->resarray)) != CUDA_SUCCESS){
			fprintf(stderr,"Error freeing result array %d (%d)\n",t->devno,cerr);
		}
		if((cerr = cuCtxPopCurrent(&t->ctx)) != CUDA_SUCCESS){
			fprintf(stderr,"Error popping context %d (%d)\n",t->devno,cerr);
		}
		if((cerr = cuCtxDestroy(t->ctx)) != CUDA_SUCCESS){
			fprintf(stderr,"Error freeing context %d (%d)\n",t->devno,cerr);
		}
		free(t);
	}
}

static int
global_cleanup(void){
	int ret = 0;

	free_devices(devices);
	free_maps(maps);
	pci_cleanup(pci);
	return ret;
}

static unsigned
max_fp_precision(const cudadev *c){
	if(c->major >= 2 || c->minor >= 3){
		return 64;
	}
	return 32;
}

static int
add_to_history(const char *rl){
	if(strcmp(rl,"") == 0){
		return 0;
	}
	add_history(rl); // libreadline has no proviso for error checking :/
	return 0;
}

static cudamap *
create_cuda_map(cudadev *dev,uintptr_t p,size_t s,void *targ){
	cudamap *r;

	if(s == 0){
		return NULL;
	}
	if((r = (cudamap *)malloc(sizeof(*r))) == NULL){
		fprintf(stderr,"Couldn't allocate map (%s)\n",strerror(errno));
		return NULL;
	}
	r->base = p;
	r->s = s;
	r->maps = targ;
	// allocno of 0 is internal; we ought only need one internal alloc.
	r->allocno = dev->alloccount++;
	return r;
}

typedef int (*cudashfxn)(const char *,const char *);

static int
cudash_quit(const char *c,const char *cmdline){
	if(strcmp(cmdline,"")){
		fprintf(stderr,"Command line following %s; did you really mean to quit?\n",c);
		return 0;
	}
	if(!cudash_child){
		printf("Thank you for using the CUDA shell. Have a very CUDA day.\n");
	}
	global_cleanup();
	exit(EXIT_SUCCESS);
}

static int
cuda_cardinfo(const cudadev *cd){
	unsigned mem,tmem;
	int attr,cerr;
	CUdevice c;

	if((cerr = cuDeviceGet(&c,cd->devno)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't associative with device %d (%d)\n",cd->devno,cerr);
		return -1;
	}
	if(cuMemGetInfo(&mem,&tmem)){
		return -1;
	}
#define MB(x) (((x) >> 20u) + !!(x % (1024 * 1024)))
	if(printf("Memory free:\t %u/%u (%u/%u MB, %2.2f%%)\n",mem,tmem,
			MB(mem),MB(tmem),(float)mem * 100 / tmem) < 0){
		return -1;
	}
#undef MB
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_INTEGRATED,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("Integrated:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("Shared maps:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("Copy+compute:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("Multikernel:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_ECC_ENABLED,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("ECC enabled:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,c);
	if(cerr != CUDA_SUCCESS || (!!attr != attr) || printf("Timelimits:\t %s\n",
				attr ? "Yes" : "No") < 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,c);
	if(cerr != CUDA_SUCCESS || printf("Compute mode:\t %s\n",
			attr == CU_COMPUTEMODE_EXCLUSIVE ? "Exclusive" :
			attr == CU_COMPUTEMODE_PROHIBITED ? "Prohibited" :
			attr == CU_COMPUTEMODE_DEFAULT ? "Shared" : "Unknown") < 0){;
		return cerr;
	}
	if(printf("Warp size:\t %d\n",cd->warpsz) < 0){
		return -1;
	}
	if(printf("Max thr/block:\t %d\n",cd->thrperblk) < 0){
		return -1;
	}
	if(printf("Max FP precis:\t %u bits\n",max_fp_precision(cd)) < 0){
		return -1;
	}
	return 0;
}

static int
list_cards(void){
	cudadev *c;

	for(c = devices ; c ; c = c->next){
		if(printf("Card %d:\t\t %s, capability %d.%d, %d MP%s\n",
			c->devno,c->devname,c->major,c->minor,
			c->mpcount,c->mpcount == 1 ? "" : "s") < 0){
			return -1;
		}
		if(kernel_cardinfo(c->devno)){
			return -1;
		}
		if(cuda_cardinfo(c)){
			return -1;
		}
		if(printf("\n") < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_cards(const char *c,const char *cmdline){
	ENFORCE_ARGEND(c,cmdline);
	return list_cards();
}

static int
create_ctx_mapofmap(cudadev *dev,uintptr_t p,size_t size,void *targ){
	cudamap *cm,**m;

	if((cm = create_cuda_map(dev,p,size,targ)) == NULL){
		return 0;
	}
	m = &dev->map;
	while(*m){
		if(cm->base <= (*m)->base){
			break;
		}
		m = &(*m)->next;
	}
	cm->next = *m;
	*m = cm;
	return 0;
}

static inline int
create_ctx_map(cudadev *dev,uintptr_t p,size_t size){
	return create_ctx_mapofmap(dev,p,size,MAP_FAILED);
}

__global__ void
clockkernel(uint64_t clocks){
	__shared__ typeof(clocks) p[GRID_SIZE * BLOCK_SIZE];
	uint64_t c0 = clock();

	p[blockIdx.x * blockDim.x + threadIdx.x] = 0;
	while(clocks >= 0x100000000ull){
		c0 = clock();
		do{
			++p[blockIdx.x * blockDim.x + threadIdx.x];
		}while(clock() > c0);
		while(clock() < c0){
			++p[blockIdx.x * blockDim.x + threadIdx.x];
		}
		clocks -= 0x100000000ull;
	}
	c0 = clock();
	do{
		++p[blockIdx.x * blockDim.x + threadIdx.x];
	}while(clock() - c0 < clocks);
}

static int
get_resarray(CUdeviceptr *r,size_t *s){
	CUresult cerr;

	*s = sizeof(uint32_t) * BLOCK_SIZE * GRID_SIZE;
	if((cerr = cuMemAlloc(r,*s)) != CUDA_SUCCESS){
		unsigned flags = CU_MEMHOSTALLOC_DEVICEMAP;
		void *vr;

		printf("Falling back to host allocation for result array...\n");
		if((cerr = cuMemHostAlloc(&vr,*s,flags)) != CUDA_SUCCESS){
			fprintf(stderr,"Couldn't allocate result array (%d)\n",cerr);
			return -1;
		}
		if((cerr = cuMemHostGetDevicePointer(r,vr,curdev->devno)) != CUDA_SUCCESS){
			fprintf(stderr,"Couldn't map result array to dev %d (%d)\n",
					curdev->devno,cerr);
			cuMemFreeHost(vr);
			return -1;
		}
	}
	if((cerr = cuMemsetD32(*r,0,*s / sizeof(uint32_t))) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize result array (%d)\n",cerr);
		cuMemFree(*r);
		return -1;
	}
	return 0;
}

static int
cudash_read(const char *c,const char *cmdline){
	uint32_t hostres[BLOCK_SIZE * GRID_SIZE];
	unsigned long long base,size;
	dim3 db(BLOCK_SIZE,1,1);
	dim3 dg(GRID_SIZE,1,1);
	CUdeviceptr res;
	CUresult cerr;
	char *ep;

	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid base: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	cmdline = ep;
	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid size: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	if(printf("Reading [0x%llx:0x%llx) (0x%llx)\n",base,base + size,size) < 0){
		return -1;
	}
	res = curdev->resarray;
	if((cerr = cuMemsetD32(res,0,BLOCK_SIZE * GRID_SIZE)) != CUDA_SUCCESS){
		if(fprintf(stderr,"Couln't initialize result array (%d)\n",cerr) < 0){
			return -1;
		}
	}
	readkernel<<<dg,db>>>((unsigned *)base,(unsigned *)(base + size),
				(uint32_t *)res);
	if((cerr = cuMemcpyDtoH(hostres,res,sizeof(hostres))) != CUDA_SUCCESS){
		if(fprintf(stderr,"Error reading memory (%d)\n",cerr) < 0){
			return -1;
		}
	}else{
		uintmax_t csum = 0;
		unsigned i;

		for(i = 0 ; i < sizeof(hostres) / sizeof(*hostres) ; ++i){
			csum += hostres[i];
		}
		if(printf("Successfully read memory (checksum: 0x%016jx (%ju)).\n",csum,csum) < 0){
			return -1;
		}
	}
	return 0;
}

__global__ void
writekernel(unsigned *aptr,const unsigned *bptr,unsigned val,uint32_t *results){
	__shared__ typeof(*results) psum[GRID_SIZE * BLOCK_SIZE];

	psum[blockDim.x * blockIdx.x + threadIdx.x] =
		results[blockDim.x * blockIdx.x + threadIdx.x];
	while(aptr + blockDim.x * blockIdx.x + threadIdx.x < bptr){
		if( (aptr[blockDim.x * blockIdx.x + threadIdx.x] = val) ){
			++psum[blockDim.x * blockIdx.x + threadIdx.x];
		}
		++psum[blockDim.x * blockIdx.x + threadIdx.x];
		aptr += blockDim.x * gridDim.x;
	}
	results[blockDim.x * blockIdx.x + threadIdx.x] =
		psum[blockDim.x * blockIdx.x + threadIdx.x];
}

static int
cudash_memset(const char *c,const char *cmdline){
	unsigned long long base,size,val;
	CUresult cerr;
	char *ep;

	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid base: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	cmdline = ep;
	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid size: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	cmdline = ep;
	if(((val = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid wvalue: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if(printf("Writing [0x%llx:0x%llx) (0x%llx)\n",base,base + size,size) < 0){
		return -1;
	}
	if((cerr = cuMemsetD8(base,val,size)) != CUDA_SUCCESS){
		if(fprintf(stderr,"Error writing memory (%d)\n",cerr) < 0){
			return -1;
		}
	}else{
		if(printf("Successfully wrote memory.\n") < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_write(const char *c,const char *cmdline){
	uint32_t hostres[BLOCK_SIZE * GRID_SIZE];
	unsigned long long base,size,val;
	dim3 db(BLOCK_SIZE,1,1);
	dim3 dg(GRID_SIZE,1,1);
	CUdeviceptr res;
	CUresult cerr;
	char *ep;

	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid base: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	cmdline = ep;
	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		if(fprintf(stderr,"Invalid size: %s\n",cmdline) < 0){
			return -1;
		}
		return 0;
	}
	cmdline = ep;
	if(((val = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid wvalue: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	res = curdev->resarray;
	if(printf("Writing [0x%llx:0x%llx) (0x%llx)\n",base,base + size,size) < 0){
		return -1;
	}
	if((cerr = cuMemsetD32(res,0,BLOCK_SIZE * GRID_SIZE)) != CUDA_SUCCESS){
		if(fprintf(stderr,"Couln't initialize result array (%d)\n",cerr) < 0){
			return -1;
		}
	}
	writekernel<<<dg,db>>>((unsigned *)base,(unsigned *)(base + size),
				val,(uint32_t *)res);
	if((cerr = cuMemcpyDtoH(hostres,res,sizeof(hostres))) != CUDA_SUCCESS){
		if(fprintf(stderr,"Error writing memory (%d)\n",cerr) < 0){
			return -1;
		}
	}else{
		uintmax_t csum = 0;
		unsigned i;

		for(i = 0 ; i < sizeof(hostres) / sizeof(*hostres) ; ++i){
			csum += hostres[i];
		}
		if(printf("Successfully wrote memory (verify: 0x%016jx (%ju)).\n",csum,csum) < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_clocks(const char *c,const char *cmdline){
	unsigned long long clocks;
	dim3 db(BLOCK_SIZE,1,1);
	dim3 dg(GRID_SIZE,1,1);
	CUresult cerr;
	char *ep;

	if(((clocks = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid clocks: %s\n",cmdline);
		return 0;
	}
	clockkernel<<<dg,db>>>(clocks);
	if((cerr = cuCtxSynchronize()) != CUDA_SUCCESS){
		if(fprintf(stderr,"Error spinning on device (%d)\n",cerr) < 0){
			return -1;
		}
	}
	printf("Occupied %llu clocks\n",clocks);
	return 0;
}

static int
cudash_alloc(const char *c,const char *cmdline){
	unsigned long long size;
	CUdeviceptr p;
	CUresult cerr;
	char *ep;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	if((cerr = cuMemAlloc(&p,size)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't allocate %llub (%d)\n",size,cerr);
		return 0;
	}
	if(create_ctx_map(curdev,p,size)){
		cuMemFree(p);
		return 0;
	}
	printf("Allocated %llub @ %p\n",size,p);
	return 0;
}

typedef struct cudafl {
	uintmax_t ptr,s;
	struct cudafl *next;
} cudafl; // cuda free-list

static void
free_cudaflist(cudafl *h){
	while(h){
		CUresult cerr;

		cudafl *t = h;

		h = h->next;
		if((cerr = cuMemFree(t->ptr)) != CUDA_SUCCESS){
			fprintf(stderr,"Warning: couldn't free %ju @ %ju (%d)\n",
			       t->s,t->ptr,cerr);
		}
		free(t);
	}
}

static int
cudash_allocat(const char *c,const char *cmdline){
	unsigned long long size,addr;
	CUdeviceptr p = 0x101200; // FIXME massive hack
	cudafl *head,**chain;
	CUresult cerr;
	char *ep;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if(((addr = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid addr: %s\n",cmdline);
		return 0;
	}
	ENFORCE_ARGEND(c,ep);
	if(addr % MINIMUM_ALLOC){
		fprintf(stderr,"Insufficiently aligned: 0x%llx\n",addr);
		return 0;
	}
	head = NULL;
	chain = NULL;
	while(1){
		cudafl *t;

		if(p != addr){
			if(p > addr){
				fprintf(stderr,"Couldn't place %llub at 0x%llx (got 0x%llx)\n",size,addr,p);
				free_cudaflist(head);
				return 0;
			}else if(p + size > addr){
				printf("fillin %llu 0x%llx\n",addr - p,addr - p);
				if((cerr = cuMemAlloc(&p,addr - p)) != CUDA_SUCCESS){
					fprintf(stderr,"Couldn't allocate %llub (%d)\n",addr - p,cerr);
					free_cudaflist(head);
					return 0;
				}
			}
		}
		// FIXME need do larger allocations to span voids
		printf("gofor %llu 0x%llx\n",size,size);
		if((cerr = cuMemAlloc(&p,size)) != CUDA_SUCCESS){
			fprintf(stderr,"Couldn't allocate %llub (%d)\n",size,cerr);
			free_cudaflist(head);
			return 0;
		}
		printf("got %llu 0x%llx at %llu (0x%llx)\n",size,size,p,p);
		if((t = (cudafl *)malloc(sizeof(*t))) == NULL){
			fprintf(stderr,"Couldn't allocate %zub (%s)\n",
					sizeof(*t),strerror(errno));
			free_cudaflist(head);
			return 0;
		}
		if(p == addr){
			break;
		}
		t->ptr = p;
		t->s = size;
		t->next = NULL;
		if(chain == NULL){
			head = t;
		}else{
			*chain = t;
		}
		chain = &t->next;
	}
	free_cudaflist(head);
	if(create_ctx_map(curdev,p,size)){
		cuMemFree(p);
		return 0;
	}
	printf("Allocated %llub @ %p\n",size,p);
	return 0;
}

static int
cudash_allocmax(const char *c,const char *cmdline){
	uintmax_t size;
	CUdeviceptr p;

	ENFORCE_ARGEND(c,cmdline);
	if((size = cuda_alloc_max(stdout,&p,1)) == 0){
		return 0;
	}
	if(create_ctx_map(curdev,p,size)){
		cuMemFree(p);
		return 0;
	}
	printf("Allocated %llub @ %p\n",size,p);
	return 0;
}

static int
cudash_pinmax(const char *c,const char *cmdline){
	unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
	uintmax_t size;
	CUdeviceptr cd;
	CUresult cerr;
	void *p;

	ENFORCE_ARGEND(c,cmdline);
	if((size = cuda_hostalloc_max(stdout,&p,1,flags)) == 0){
		return 0;
	}
	printf("Allocated %llub host memory @ %p\n",size,p);
	// FIXME map into each card's memory space, not just current's
	if((cerr = cuMemHostGetDevicePointer(&cd,p,curdev->devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't map %llub @ %p on dev %d (%d)\n",
				size,p,curdev->devno,cerr);
		cuMemFreeHost(p);
		return 0;
	}
	printf("Mapped %llub into card %d @ %p\n",size,0,cd);
	if(create_ctx_map(&systemdev,(uintptr_t)p,size)){
		cuMemFreeHost(p);
		return 0;
	}
	if(create_ctx_mapofmap(curdev,cd,size,p)){
		cuMemFreeHost(p);
		// FIXME need to extract from host map list
		return 0;
	}
	return 0;
}

static int
cudash_pin(const char *c,const char *cmdline){
	unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
	unsigned long long size;
	CUdeviceptr cd;
	CUresult cerr;
	char *ep;
	void *p;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	ENFORCE_ARGEND(c,ep);
	if((cerr = cuMemHostAlloc(&p,size,flags)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't host-allocate %llub (%d)\n",size,cerr);
		return 0;
	}
	printf("Allocated %llub host memory @ %p\n",size,p);
	if((cerr = cuMemHostGetDevicePointer(&cd,p,curdev->devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't map %llub @ %p on dev %d (%d)\n",
				size,p,curdev->devno,cerr);
		cuMemFreeHost(p);
		return 0;
	}
	printf("Mapped %llub into card %d @ %p\n",size,0,cd);
	if(create_ctx_map(&systemdev,(uintptr_t)p,size)){
		cuMemFreeHost(p);
		return 0;
	}
	if(create_ctx_mapofmap(curdev,cd,size,p)){
		cuMemFreeHost(p);
		// FIXME need to extract from host map list
		return 0;
	}
	return 0;
}

static int
cudash_fork(const char *c,const char *cmdline){
	pid_t pid;

	ENFORCE_ARGEND(c,cmdline);
	if(fflush(stdout) || fflush(stderr)){
		fprintf(stderr,"Couldn't flush output (%s?)\n",strerror(errno));
		return -1;
	}
	if((pid = fork()) < 0){
		fprintf(stderr,"Couldn't fork (%s?)\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		cudash_child = 1;
		printf("Type \"exit\" to leave this child (PID %ju)\n",(uintmax_t)getpid());
		return 0;
	}else{
		int status;

		while(waitpid(pid,&status,0) != pid){
			if(errno != EINTR){
				fprintf(stderr,"Couldn't wait for child %ju (%s)\n",
					(uintmax_t)pid,strerror(errno));
				return -1;
			}
		}
		if(!WIFEXITED(status) || WEXITSTATUS(status)){
			fprintf(stderr,"Child %ju terminated abnormally, exiting\n",
					(uintmax_t)pid);
			return -1;
		}
		printf("Returning to parent shell (PID %ju)\n",(uintmax_t)getpid());
	}
	return 0;
}

static int
cudash_free(const char *c,const char *cmdline){
	unsigned long long base,size;
	cudamap **m;
	char *ep;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid base: %s\n",cmdline);
		return 0;
	}
	ENFORCE_ARGEND(c,ep);
	m = &curdev->map;
	while(*m){
		CUresult cerr;
		cudamap *tmp;

		if(!(*m)->allocno){ // don't free internalallocs
			m = &(*m)->next;
			continue;
		}
		// The rule here is that we don't free unless the to-free span
		// completely covers the allocation...is that what we want?
		if((*m)->base < base || base + size < (*m)->base + (*m)->s){
			m = &(*m)->next;
			continue;
		}
		if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx",
			curdev->devno,(*m)->s,(*m)->s,(uintmax_t)(*m)->base) < 0){
			return -1;
		}
		if((*m)->maps != MAP_FAILED){
			if(printf(" maps %12p",(*m)->maps) < 0){
				return -1;
			}
			fprintf(stderr,"Freeing mappings is not yet implemented\n");
			return -1; // FIXME not yet supported...
		}else{
			cerr = cuMemFree((*m)->base);
		}
		if(printf("\n") < 0){
			return -1;
		}
		if(cerr){
			if(fprintf(stderr,"Error freeing device region (%d)\n",cerr)){
				return -1;
			}
		}
		tmp = *m;
		*m = (*m)->next;
		free(tmp);
	}
	return 0;
}

static int
cudash_exec(const char *c,const char *cmdline){
	pid_t pid;

	if(fflush(stdout) || fflush(stderr)){
		fprintf(stderr,"Couldn't flush output (%s?)\n",strerror(errno));
		return -1;
	}
	if((pid = fork()) < 0){
		fprintf(stderr,"Couldn't fork (%s?)\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		while(isspace(*cmdline)){
			++cmdline;
		}{
			// FIXME tokenize that bitch
			char * const argv[] = { strdup(cmdline), NULL };
			if(execvp(cmdline,argv)){
				fprintf(stderr,"Couldn't launch %s (%s)\n",cmdline,strerror(errno));
			}
		}
		exit(EXIT_FAILURE);
	}else{
		int status;

		while(waitpid(pid,&status,0) != pid){
			if(errno != EINTR){
				fprintf(stderr,"Couldn't wait for child %ju (%s)\n",
					(uintmax_t)pid,strerror(errno));
				return -1;
			}
		}
		if(WIFEXITED(status)){
			if(WEXITSTATUS(status)){
				printf("Child %ju terminated with status %d\n",
						(uintmax_t)pid,WEXITSTATUS(status));
			}
		}else if(WIFSIGNALED(status)){
			printf("Child %ju killed with signal %d\n",
					(uintmax_t)pid,WTERMSIG(status));
		}
	}
	return 0;
}

static int
cudash_maps(const char *c,const char *cmdline){
	cudadev *d;
	cudamap *m;

	ENFORCE_ARGEND(c,cmdline);
	for(m = maps ; m ; m = m->next){
		if(printf("(host) %10zu (0x%08x) @ 0x%012jx\n",
				m->s,m->s,(uintmax_t)m->base) < 0){
			return -1;
		}
	}
	for(d = devices ; d ; d = d->next){
		uintmax_t b = 0;

		for(m = d->map ; m ; m = m->next){
			if((uintmax_t)m->base > b){
				uintmax_t skip = (uintmax_t)m->base - b;

				if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx unallocated\n",
						d->devno,skip,skip,b) < 0){
					return -1;
				}
			}
			if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx",
					d->devno,m->s,m->s,(uintmax_t)m->base) < 0){
				return -1;
			}
			if(m->maps != MAP_FAILED){
				if(printf(" maps %12p",m->maps) < 0){
					return -1;
				}
			}else if(m->allocno){
				if(printf(" user alloc #%u",m->allocno) < 0){
					return -1;
				}
			}else{
				if(printf(" cudash result buffer",m->maps) < 0){
					return -1;
				}
			}
			if(printf("\n") < 0){
				return -1;
			}
			b = (uintmax_t)m->base + m->s;
		}
		if(b != (1ull << d->addrbits)){
			uintmax_t skip = (1ull << d->addrbits) - b;

			if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx unallocated\n",
					d->devno,skip,skip,b) < 0){
				return -1;
			}
			b += skip;
		}
	}
	return 0;
}

static int
cudash_verify(const char *c,const char *cmdline){
	uint32_t hostres[BLOCK_SIZE * GRID_SIZE];
	dim3 db(BLOCK_SIZE,1,1);
	dim3 dg(GRID_SIZE,1,1);
	CUdeviceptr res;
	CUresult cerr;
	cudadev *d;
	cudamap *m;

	ENFORCE_ARGEND(c,cmdline);
	res = curdev->resarray;
	for(d = devices ; d ; d = d->next){
		for(m = d->map ; m ; m = m->next){
			if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx",
					d->devno,m->s,m->s,(uintmax_t)m->base) < 0){
				return -1;
			}
			if(m->maps != MAP_FAILED){
				if(printf(" (maps %12p)",m->maps) < 0){
					return -1;
				}
			}
			if(printf("\n") < 0){
				return -1;
			}
			if((cerr = cuMemsetD32(res,0,BLOCK_SIZE * GRID_SIZE)) != CUDA_SUCCESS){
				if(fprintf(stderr,"Couln't initialize result array (%d)\n",cerr) < 0){
					return -1;
				}
			}
			readkernel<<<dg,db>>>((unsigned *)m->base,(unsigned *)(m->base + m->s),
						(uint32_t *)res);
			if((cerr = cuMemcpyDtoH(hostres,res,sizeof(hostres))) != CUDA_SUCCESS){
				if(fprintf(stderr,"Error reading memory (%d)\n",cerr) < 0){
					return -1;
				}
			}else{
				uintmax_t csum = 0;
				unsigned i;

				for(i = 0 ; i < sizeof(hostres) / sizeof(*hostres) ; ++i){
					csum += hostres[i];
				}
				if(printf(" Successfully read memory (checksum: 0x%016jx (%ju)).\n",csum,csum) < 0){
					return -1;
				}
			}
		} 
	}
	return 0;
}

static int
cudash_wverify(const char *c,const char *cmdline){
	uint32_t hostres[BLOCK_SIZE * GRID_SIZE];
	dim3 db(BLOCK_SIZE,1,1);
	unsigned long long val;
	dim3 dg(GRID_SIZE,1,1);
	CUdeviceptr res;
	CUresult cerr;
	cudadev *d;
	cudamap *m;
	char *ep;

	if(((val = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid value: %s\n",cmdline);
		return 0;
	}
	ENFORCE_ARGEND(c,ep);
	cmdline = ep;
	res = curdev->resarray;
	for(d = devices ; d ; d = d->next){
		for(m = d->map ; m ; m = m->next){
			if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx",
					d->devno,m->s,m->s,(uintmax_t)m->base) < 0){
				return -1;
			}
			if(m->maps != MAP_FAILED){
				if(printf(" (maps %12p)",m->maps) < 0){
					return -1;
				}
			}
			if(printf("\n") < 0){
				return -1;
			}
			if((cerr = cuMemsetD32(res,0,BLOCK_SIZE * GRID_SIZE)) != CUDA_SUCCESS){
				if(fprintf(stderr,"Couln't initialize result array (%d)\n",cerr) < 0){
					return -1;
				}
			}
			writekernel<<<dg,db>>>((unsigned *)m->base,(unsigned *)(m->base + m->s),
						val,(uint32_t *)res);
			if((cerr = cuMemcpyDtoH(hostres,res,sizeof(hostres))) != CUDA_SUCCESS){
				if(fprintf(stderr,"Error writing memory (%d)\n",cerr) < 0){
					return -1;
				}
			}else{
				uintmax_t csum = 0;
				unsigned i;

				for(i = 0 ; i < sizeof(hostres) / sizeof(*hostres) ; ++i){
					csum += hostres[i];
				}
				if(printf(" Successfully wrote memory (checksum: 0x%016jx (%ju)).\n",csum,csum) < 0){
					return -1;
				}
			}
		}
	}
	return 0;
}

#define CUCTXSIZE 48 // FIXME just a guess

static int
list_contexts(void){
	cudadev *c;

	for(c = devices ; c ; c = c->next){
		CUcontext ctx = c->ctx;
		unsigned z;

		if(printf("Card %d:\t\t %s, capability %d.%d, %d MP%s\n",
			c->devno,c->devname,c->major,c->minor,c->mpcount,
			c->mpcount == 1 ? "" : "s") < 0){
			return -1;
		}
		for(z = 0 ; z < CUCTXSIZE ; ++z){
			if(printf(" %02x",((const unsigned char *)ctx)[z]) < 0){
				return -1;
			}
		}
		if(printf(" (%p)\n",ctx) < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_ctxdump(const char *c,const char *cmdline){
	ENFORCE_ARGEND(c,cmdline);
	return list_contexts();
}

static int
cudash_device(const char *c,const char *cmdline){
	unsigned long long devno;
	cudadev *d;
	char *ep;

	if(((devno = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid devno: %s\n",cmdline);
		return 0;
	}
	ENFORCE_ARGEND(c,ep);
	for(d = devices ; d ; d = d->next){
		if(d->devno == devno){
			curdev = d;
			return 0;
		}
	}
	if(fprintf(stderr,"%llu is not a valid device number.\n",devno) < 0){
		return -1;
	}
	return 0;
}

static cudadev *
getdev(unsigned devno){
	cudadev *d;

	for(d = devices ; d ; d = d->next){
		if(d->devno == devno){
			break;
		}
	}
	return d;
}

static int
cudash_registry(const char *c,const char *cmdline){
	ENFORCE_ARGEND(c,cmdline);
	if(kernel_registry()){
		return -1;
	}
	return 0;
}

static int
library_versions(void){
	int attr,cerr,r = 0,rr;

	if((rr = printf("libpci compile version: %s\n",PCILIB_VERSION)) < 0){
		return -1;
	}
	r += rr;
	if((rr = printf("CUDA compile version: %d.%d\n",CUDAMAJMIN(CUDA_VERSION))) < 0){
		return -1;
	}
	r += rr;
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get CUDA driver version (%d)\n",cerr);
		return -1;
	}
	if((rr = printf("CUDA link version: %d.%d\n",CUDAMAJMIN(attr))) < 0){
		return -1;
	}
	r += rr;
	return r;
}

static int
cudash_driver(const char *c,const char *cmdline){
	ENFORCE_ARGEND(c,cmdline);
	if(kernel_version()){
		return -1;
	}
	if(library_versions() < 0){
		return -1;
	}
	return 0;
}

static int
cudash_pokectx(const char *c,const char *cmdline){
	unsigned long long devno,value,off;
	cudadev *cd;
	char *ep;

	if(((devno = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid devno: %s\n",cmdline);
		return 0;
	}
	if((cd = getdev(devno)) == NULL){
		fprintf(stderr,"Invalid devno: %llu\n",devno);
		return 0;
	}
	cmdline = ep;
	if(((off = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid off: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if((value = strtoull(cmdline,&ep,16)) == ULONG_MAX && errno == ERANGE
			|| cmdline == ep){
		fprintf(stderr,"Invalid value: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	ENFORCE_ARGEND(c,cmdline);
	while(off < CUCTXSIZE){
		if(value > 0xffu){
			fprintf(stderr,"Invalid value: 0x%llx\n",value);
			return 0;
		}
		((char *)cd->ctx)[off] = value;
		if(printf("Poked device %llu off %02llx with value 0x%02llx\n",
					devno,off,value) < 0){
			return -1;
		}
		++off;
		cmdline = ep;
		if((value = strtoull(cmdline,&ep,16)) == ULONG_MAX && errno == ERANGE){
			fprintf(stderr,"Invalid value: %s\n",cmdline);
			return 0;
		}
		if(cmdline == ep){
			return 0;
		}
	}
	while(isspace(*cmdline)){
		++cmdline;
	}
	if(*cmdline){
		if(fprintf(stderr,"Too much data (%s)!\n",cmdline) < 0){
			return 0;
		}
	}
	return 0;
}

static int cudash_help(const char *,const char *);

static const struct {
	const char *cmd;
	cudashfxn fxn;
	const char *help;
} cmdtable[] = {
	{ "alloc",	cudash_alloc,	"allocate device memory",	},
	{ "allocat",	cudash_allocat,	"allocate particular device memory",	},
	{ "allocmax",	cudash_allocmax,"allocate all possible contiguous device memory",	},
	{ "cards",	cudash_cards,	"list devices supporting CUDA",	},
	{ "clocks",	cudash_clocks,	"spin for a specified number of device clocks",	},
	{ "ctxdump",	cudash_ctxdump,	"serialize CUcontext objects",	},
	{ "device",	cudash_device,	"change the current device",	},
	{ "driver",	cudash_driver,	"driver version info",},
	{ "exec",	cudash_exec,	"fork, and exec a binary",	},
	{ "exit",	cudash_quit,	"exit the CUDA shell",	},
	{ "fork",	cudash_fork,	"fork a child cudash",	},
	{ "free",	cudash_free,	"free maps within a range",	},
	{ "help",	cudash_help,	"help on the CUDA shell and commands",	},
	{ "maps",	cudash_maps,	"display CUDA memory tables",	},
	{ "memset",	cudash_memset,	"write device memory from host",	},
	{ "pin",	cudash_pin,	"pin and map host memory",	},
	{ "pinmax",	cudash_pinmax,  "pin and map all possible contiguous host memory",	},
	{ "pokectx",	cudash_pokectx,	"poke values into CUcontext objects",	},
	{ "quit",	cudash_quit,	"exit the CUDA shell",	},
	{ "read",	cudash_read,	"read device memory in CUDA",	},
	{ "registry",	cudash_registry,"dump the driver's registry",	},
	{ "verify",	cudash_verify,	"verify all mapped memory is readable",	},
	{ "write",	cudash_write,	"write device memory in CUDA",	},
	{ "wverify",	cudash_wverify,	"verify all mapped memory is writeable",	},
	{ NULL,		NULL,		NULL,	}
};

static typeof(*cmdtable) *
lookup_command(const char *c,size_t n){
	typeof(*cmdtable) *tptr;

	for(tptr = cmdtable ; tptr->cmd ; ++tptr){
		if(strncmp(tptr->cmd,c,n) == 0 && strlen(tptr->cmd) == n){
			return tptr;
		}
	}
	return NULL;
}

static int
list_commands(void){
	typeof(*cmdtable) *t;

	for(t = cmdtable ; t->cmd ; ++t){
		if(printf("%s: %s\n",t->cmd,t->help) < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_help(const char *c,const char *cmdline){
	if(strcmp(cmdline,"") == 0){
		return list_commands();
	}else{
		typeof(*cmdtable) *tptr;

		while(isspace(*cmdline)){
			++cmdline;
		}
		// FIXME extract first token
		if((tptr = lookup_command(cmdline,strlen(cmdline))) == NULL){
			if(printf("No help is available for \"%s\".\n",cmdline) < 0){
				return -1;
			}
		}else{
			if(printf("%s: %s\n",tptr->cmd,tptr->help) < 0){
				return -1;
			}
		}
	}
	return 0;
}

static int
run_command(const char *cmd){
	typeof(*cmdtable) *tptr = NULL;
	struct timeval t0,t1,tsub;
	const char *toke;
	int r;

	while(isspace(*cmd)){
		++cmd;
	}
	toke = cmd;
	while(isalnum(*cmd)){
		++cmd;
	}
	if(cmd != toke){
		tptr = lookup_command(toke,cmd - toke);
	}
	if(tptr == NULL){
		if(fprintf(stderr,"Invalid command: \"%.*s\"\n",cmd - toke,toke) < 0){
			return -1;
		}
		return 0;
	}
	gettimeofday(&t0,NULL);
	if((r = tptr->fxn(tptr->cmd,cmd)) == 0){
		gettimeofday(&t1,NULL);
		timersub(&t1,&t0,&tsub);
		if(tsub.tv_sec){
			if(printf("Command took %u.%04us\n",tsub.tv_sec,tsub.tv_usec / 1000) < 0){
				return -1;
			}
		}else if(tsub.tv_usec / 1000){
			if(printf("Command took %ums\n",tsub.tv_usec / 1000) < 0){
				return -1;
			}
		}
	}
	return r;
}

static int
id_cudadev(cudadev *c){
	struct cudaDeviceProp dprop;
	size_t rsize;
	CUdevice d;
	int cerr;

	if((cerr = cuDeviceGet(&d,c->devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't query device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	if((cerr = cudaGetDeviceProperties(&dprop,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't query device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	cerr = cuDeviceGetAttribute(&c->thrperblk,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,d);
	if(cerr != CUDA_SUCCESS || c->thrperblk <= 0){
		fprintf(stderr,"Couldn't get threads/block for device %d (%d)\n",c->devno,cerr);
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&c->warpsz,CU_DEVICE_ATTRIBUTE_WARP_SIZE,d);
	if(cerr != CUDA_SUCCESS || c->warpsz <= 0){
		fprintf(stderr,"Couldn't get warp size for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	cerr = cuDeviceGetAttribute(&c->mpcount,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,d);
	if(cerr != CUDA_SUCCESS || c->mpcount <= 0){
		fprintf(stderr,"Couldn't get MP count for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	if((cerr = cuDeviceComputeCapability(&c->major,&c->minor,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get compute capability for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	if(c->major < 2){
		c->addrbits = 32;
	}else{
		c->addrbits = 40;
	}
#define CUDASTRLEN 80
	if((c->devname = (char *)malloc(CUDASTRLEN)) == NULL){
		fprintf(stderr,"Couldn't allocate %zub (%s?)\n",CUDASTRLEN,strerror(errno));
		return -1;
	}
	if((cerr = cuDeviceGetName(c->devname,CUDASTRLEN,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get name for device %d (%d)\n",c->devno,cerr);
		free(c->devname);
		return -1;
	}
#undef CUDASTRLEN
	if((cerr = cuCtxCreate(&c->ctx,CU_CTX_MAP_HOST,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get context for device %d (%d)\n",c->devno,cerr);
		free(c->devname);
		return -1;
	}
	if(get_resarray(&c->resarray,&rsize)){
		cuCtxDestroy(c->ctx);
		free(c->devname);
		return -1;
	}
	if(create_ctx_map(c,c->resarray,rsize)){
		cuCtxDestroy(c->ctx);
		free(c->devname);
		return -1;
	}
	return 0;
}

static int
make_devices(int count){
	cudadev *chain = NULL;

	while(count--){
		cudadev *c;

		if((c = (cudadev *)malloc(sizeof(*c))) == NULL){
			free_devices(chain);
			return -1;
		}
		c->alloccount = 0;
		c->devno = count;
		c->map = NULL;
		if(id_cudadev(c)){
			free_devices(chain);
			free(c);
			return -1;
		}
		c->next = chain;
		chain = c;
	}
	devices = chain;
	return 0;
}

static struct pci_access *
analyze_pci(unsigned *devs){
	struct pci_access *ret;
	struct pci_dev *d;

	*devs = 0;
	if((ret = pci_alloc()) == NULL){
		return NULL;
	}
	pci_init(ret);
	pci_scan_bus(ret);
	for(d = ret->devices ; d ; d = d->next){
		if(d->vendor_id == NVIDIA_VENDORID){
			unsigned char cspace[PCI_CONFIG_BYTES];
			char nbuf[80];
			int c;

			if(!pci_fill_info(d,PCI_FILL_CLASS)){
				fprintf(stderr,"Couldn't determine PCI class\n");
				pci_cleanup(ret);
				return NULL;
			}
			if(d->device_class & 0xff00 != PCI_VGA_CLASS){
				continue;
			}
			if(!pci_read_block(d,0,cspace,sizeof(cspace))){
				fprintf(stderr,"Couldn't read PCI configuration space\n");
				pci_cleanup(ret);
				return NULL;
			}
			c = cspace[PCI_REVISION_ID];
			printf("nVidia PCI device %04x: Bus %02x, Dev %02x, Func %02x, Rev %02x, IRQ: %u\n",
					d->device_id,d->bus,d->dev,d->func,c,d->irq);
			if(!pci_lookup_name(ret,nbuf,sizeof(nbuf),
					PCI_LOOKUP_VENDOR | PCI_LOOKUP_DEVICE,
					d->vendor_id,d->device_id)){
				fprintf(stderr,"Couldn't get PCI device name\n");
				pci_cleanup(ret);
				return NULL;
			}
			printf("\t%s\n",nbuf);
			++*devs;
		}
	}
	if(*devs == 0){
		fprintf(stderr,"Warning: Didn't find any PCI nVidia display controllers...\n");
	}else{
		printf("Found %u PCI nVidia display controller%s\n",*devs,
				*devs == 1 ? "" : "s");
	}
	return ret;
}

int main(int argc,char **argv){
	const char *prompt = "cudash> ";
	unsigned pcicount;
	char *rln = NULL;
	int count;

	printf("The CUDA shell (C) Nick Black 2010. Compiled against libpci version %s.\n",PCILIB_VERSION);
	if((pci = analyze_pci(&pcicount)) == NULL){
		fprintf(stderr,"Couldn't initialize libpci\n");
		exit(EXIT_FAILURE);
	}
	if(init_cuda_alldevs(&count)){
		goto err;
	}
	if(make_devices(count)){
		goto err;
	}
	curdev = devices;

	if(argc > 1){
		// FIXME generate string from all of argv
		if(run_command(argv[1])){ // FIXME ret exact result code
			goto err;
		}
	}else{
		using_history(); // Set up GNU readline history.
		if(read_history(HISTORY_FILE)){
			// FIXME no history file for you! oh well
		}
		while( (rln = readline(prompt)) ){
			// An empty string ought neither be saved to history nor run.
			if(strcmp("",rln)){
				if(add_to_history(rln)){
					fprintf(stderr,"Error adding input to history. Exiting.\n");
					free(rln);
					goto err;
				}
				if(run_command(rln)){
					free(rln);
					goto err;
				}
			}
			free(rln);
		}
		if(write_history(HISTORY_FILE)){
			fprintf(stderr,"Warning: couldn't write history file at %s\n",HISTORY_FILE);
		}
	}
	if(global_cleanup()){
		fprintf(stderr,"Error cleaning up. Exiting.\n");
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
	
err:
	global_cleanup();
	exit(EXIT_FAILURE);
}
