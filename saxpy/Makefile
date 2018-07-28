CC=nvcc
DIR=bin

all: build

build: $(DIR)/saxpy

$(DIR)/saxpy: saxpy.cu gpu_error.cuh
	@mkdir -p $(DIR)
	$(CC) -o $(DIR)/saxpy saxpy.cu

clean:
	@rm -f $(DIR)/saxpy

run: $(DIR)/saxpy
	@$(DIR)/saxpy
