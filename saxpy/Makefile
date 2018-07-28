CC=nvcc
DIR=bin
IDIR=../common
CFLAGS=-I$(IDIR)

all: build

build: $(DIR)/saxpy

$(DIR)/saxpy: saxpy.cu $(IDIR)/gpu_error.cuh
	@mkdir -p $(DIR)
	$(CC) -o $(DIR)/saxpy saxpy.cu $(CFLAGS)

clean:
	@rm -f $(DIR)/saxpy

run: $(DIR)/saxpy
	@$(DIR)/saxpy
