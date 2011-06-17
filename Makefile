.DELETE_ON_ERROR:
.PHONY: all bin ptx profile test fulltest clean install uninstall
.DEFAULT_GOAL:=all

LOCALMAKE:=Makefile.local
include $(LOCALMAKE)

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

TAGBIN?=ctags
CUDADIR?=/usr/
CUDAINC?=$(CUDADIR)/include
CUDARTLIB?=$(CUDADIR)/lib64

TARGET?=/usr/local
TARGBIN?=$(TARGET)/bin

INSTALL?=install
NVCC?=$(CUDADIR)/bin/nvcc
GPUARCH?=compute_10
GPUCODE?=sm_12,sm_10
# FIXME restore -Werror!
CFLAGS+=-O2 -Wall -W -Wextra -march=native -mtune=native -I$(SRC) -I$(CUDAINC)
NCFLAGS+=-O2 --compiler-options -W,-Wall,-Wextra,-march=native,-mtune=native
NCFLAGS+=-arch $(GPUARCH) -code $(GPUCODE) --ptxas-options=-v -I$(SRC) -I$(CUDAINC)
LFLAGS:=--as-needed
NLFLAGS:=--linker-options $(LFLAGS),-R$(CUDARTLIB) -L$(CUDARTLIB) -lcuda -lcudart
LFLAGS:=$(addprefix -Wl,$(LFLAGS)) -lcuda -lcudart
PTXFLAGS:=--ptx
TAGS:=.tags

all: $(TAGS) bin ptx
       
bin: $(TAGS) $(BIN)
	
ptx: $(PTX)

$(TAGS): $(CSRC) $(CUDASRC) util/cubar.c $(SRC)/cubar.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(TAGBIN) --langmap=c:.c.cu.h -f $@ $^

$(CUDAMINIMAL): $(OUT)/cudaminimal.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(CUDAPINNER): $(OUT)/cudapinner.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(CUDASPAWNER): $(OUT)/cudaspawner.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(CUDASTUFFER): $(OUT)/cudastuffer.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(CUDABOUNDER): $(OUT)/cudabounder.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/%: $(OUT)/%.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudash: $(OUT)/cudash.o $(OUT)/cubar.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS) -lreadline -lpci

$(OUT)/%.ptx: $(SRC)/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(PTXFLAGS) $(NCFLAGS) -o $@ $<

$(OUT)/%.o: $(SRC)/%.cu $(SRC)/cubar.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -c -o $@ $<

$(OUT)/%.o: $(SRC)/%.c $(SRC)/cubar.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -c -o $@ $<

$(OUT)/%.o: util/%.c $(SRC)/cubar.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -c -o $@ $<

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
	# This fails on 195.xx.xx, but succeeds on 256.xx....
	#./$(CUDARANGER) $(CUDADEVNO) 0 0x1000

fulltest: test
	./$(CUDADUMP)
	./$(CUDAPINNER) 0
	./$(CUDAQUIRKY)
	./$(CUDARANGER) $(CUDADEVNO) 0 0x100000000

clean:
	rm -rf out $(TAGS) $(wildcard *.dump)

install:
	$(INSTALL) -d $(TARGBIN)
	$(INSTALL) $(BIN) $(TARGBIN)

uninstall:
	rm -rf $(addprefix $(TARGBIN),$(BIN))

$(LOCALMAKE):
	@[ -d $(@D) ] || mkdir -p $(@D)
	touch $@
