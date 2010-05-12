.DELETE_ON_ERROR:
.PHONY: all bin ptx profile test fulltest clean
.DEFAULT_GOAL:=test

include Makefile.local

SRC:=src
OUT:=out
CUDABOUNDER:=out/cudabounder
CUDADUMP:=out/cudadump
CUDAMINIMAL:=out/cudaminimal
CUDAPINNER:=out/cudapinner
CUDAQUIRKY:=out/cudaquirky
CUDARANGER:=out/cudaranger
CUDASH:=out/cudash
CUDASPAWNER:=out/cudaspawner
CUDASTUFFER:=out/cudastuffer
CSRC:=$(wildcard src/*.c)
CUDASRC:=$(wildcard src/*.cu)
BIN:=$(addprefix out/,$(basename $(notdir $(CSRC) $(CUDASRC))))
PTX:=$(addsuffix .ptx,$(addprefix out/,$(basename $(notdir $(CUDASRC)))))
PROFDATA:=$(addsuffix .prof,$(BIN))

CUDA?=/usr/
CUDAINC?=$(CUDA)/include/cuda
CUDART?=$(HOME)/local/cuda/
CUDARTLIB:=$(CUDART)/lib64

NVCC?=$(CUDA)/bin/nvcc
GPUARCH?=compute_10
GPUCODE?=sm_12,sm_10
CFLAGS:=-O2 -Wall -W -Werror -march=native -mtune=native
NCFLAGS:=-O2 --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(SRC) -I$(CUDAINC)
NCFLAGS+=-arch $(GPUARCH) -code $(GPUCODE) --compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=-lcuda
NLFLAGS:=$(LFLAGS) --linker-options -R$(CUDARTLIB)
PTXFLAGS:=--ptx
TAGS:=.tags

all: $(TAGS) bin ptx
       
bin: $(TAGS) $(BIN)
	
ptx: $(PTX)

$(TAGS): $(CSRC) $(CUDASRC) util/cuda8803ss.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	ctags --langmap=c:.c.cu.h -f $@ $^

$(OUT)/cudabounder: $(OUT)/cudabounder.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudadump: $(OUT)/cudadump.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudaminimal: $(OUT)/cudaminimal.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudapinner: $(OUT)/cudapinner.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudaquirky: $(OUT)/cudaquirky.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudaranger: $(OUT)/cudaranger.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudash: $(OUT)/cudash.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS) -lreadline

$(OUT)/cudaspawner: $(OUT)/cudaspawner.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudastuffer: $(OUT)/cudastuffer.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/%.ptx: $(SRC)/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(PTXFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -c -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) -I$(SRC) -I$(CUDAINC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

$(OUT)/%.o: util/%.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) -I$(SRC) -I$(CUDAINC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

profile: $(PROFDATA)

PROF:=CUDA_PROFILE_LOG=$(shell pwd)/$(PROFDATA) CUDA_PROFILE=1
$(PROFDATA): test
	@[ -d $(@D) ] || mkdir -p $(@D)
	env $(PROF) out/cudadump
	cat $@

CUDADEVNO?=0
test: bin
	./$(CUDARANGER) $(CUDADEVNO) 0x1000 0x201000
	./$(CUDASTUFFER) $(CUDADEVNO)
	./$(CUDASPAWNER) $(CUDADEVNO) 0x100000
	! ./$(CUDARANGER) $(CUDADEVNO) 0 0x1000

fulltest: test
	./$(CUDADUMP)
	./$(CUDAPINNER) 0
	./$(CUDAQUIRKY)
	! ./$(CUDARANGER) $(CUDADEVNO) 0 0x100000000

clean:
	rm -rf out $(TAGS) $(wildcard *.dump)
