CC=nvcc
DIR=bin
IDIR=../common
CFLAGS=-I$(IDIR)

all: build

build: $(DIR)/stride

$(DIR)/stride: stride.cu $(IDIR)/gpu_error.cuh
	@mkdir -p $(DIR)
	$(CC) -o $(DIR)/stride stride.cu $(CFLAGS)

clean:
	@rm -f $(DIR)/stride

run: $(DIR)/stride
	@$(DIR)/stride

profile: build
	@nvprof $(DIR)/stride
