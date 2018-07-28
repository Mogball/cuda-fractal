all: main

temp.o: mandelbrot.cu
	nvcc -rdc=true -c -o temp.o mandelbrot.cu -I ../include

mandelbrot.o: temp.o
	nvcc -dlink -o mandelbrot.o temp.o -l cudart

libgpu.a: mandelbrot.o temp.o
	ar cr libgpu.a mandelbrot.o temp.o
	ranlib libgpu.a

main: main.cpp libgpu.a
	g++ main.cpp -L ./ -l gpu -o main -L /usr/local/cuda/lib64 -l cudart -L ../bitmap/bin -l bitmap --std=c++11 -I ../include

run: main
	./main

