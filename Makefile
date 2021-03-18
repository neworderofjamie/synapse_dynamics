
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_50 -std=c++11 -O3

all: synapse_dynamics

synapse_dynamics: kernel.o
	$(CXX) -o synapse_dynamics kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart
	
kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f synapse_dynamics kernel.o
