CC:=mpic++
NVCC:=nvcc
LDFLAGS:=-L$(CUDA_ROOT)/lib64 -L$(MPI_ROOT)/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX=
CFLAGS:=-std=c++11 -I$(MPI_ROOT)/include -I. -I$(CUDA_ROOT)/include -DOMPI_SKIP_MPICXX=

RING_EXE_NAME:=allreduce-ring-test
DOUBLE_TREE_EXE_NAME:=allreduce-double-tree-test
RECURSIVE_EXE_NAME:=allreduce-recursive-test

SRC:=$(wildcard *.cpp test/*.cpp)
CU_SRC:=$(wildcard *.cu)
OBJS:=$(SRC:.cpp=.o) $(CU_SRC:.cu=.o)

all: $(RING_EXE_NAME)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@

$(RING_EXE_NAME): collectives.o test/test.o timer.o
	$(CC) -o $@ $(LDFLAGS) $^ $(LDFLAGS)

#$(DOUBLE_TREE_EXE_NAME): double_tree_collectives.o test/double_tree_test.o timer.o
#	$(CC) -o $@ $(LDFLAGS) $^ $(LDFLAGS)
#
#$(RECURSIVE_EXE_NAME): rchr_collectives.o test/rchr_test.o timer.o
#	$(CC) -o $@ $(LDFLAGS) $^ $(LDFLAGS)

ring-test: $(RING_EXE_NAME)
	$<

#dbtree-test: $(DOUBLE_TREE_EXE_NAME)
#	$<
#
#rchr-test: $(RECURSIVE_EXE_NAME)
#	$<

clean:
	rm -f *.o test/*.o $(RING_EXE_NAME) $(DOUBLE_TREE_EXE_NAME) $(RECURSIVE_EXE_NAME)
