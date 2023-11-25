CC:=mpic++
NVCC:=nvcc
LDFLAGS:=-L$(CUDA_ROOT)/lib64 -L$(MPI_ROOT)/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX=
CFLAGS:=-std=c++11 -I$(MPI_ROOT)/include -I. -I$(CUDA_ROOT)/include -DOMPI_SKIP_MPICXX=

RING_EXE_NAME:=allreduce-ring-test
RH_EXE_NAME:=allreduce-recursive-test

SRC:=$(wildcard *.cpp test/*.cpp)
CU_SRC:=$(wildcard *.cu)
OBJS:=$(SRC:.cpp=.o) $(CU_SRC:.cu=.o)

all: $(RING_EXE_NAME)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@

$(RING_EXE_NAME): collectives.o timer.o test/test.o
	$(CC) -o $(RING_EXE_NAME) $(LDFLAGS) $^ $(LDFLAGS)

$(RH_EXE_NAME): rh_collectives.o timer.o test/rh_test.o
	$(CC) -o $(RH_EXE_NAME) $(LDFLAGS) $^ $(LDFLAGS)

test-ring: $(RING_EXE_NAME)
	$(RING_EXE_NAME)

test-rh: $(RH_EXE_NAME)
	$(RH_EXE_NAME)

clean:
	rm -f *.o test/*.o $(EXE_NAME)
