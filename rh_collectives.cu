#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>

#include "collectives.h"

struct MPIGlobalState {
    // The CUDA device to run on, or -1 for CPU-only.
    int device = -1;

    // A CUDA stream (if device >= 0) initialized on the device
    cudaStream_t stream;

    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
};

// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void RecursiveInitCollectives(int device) {
    if(device < 0) {
        // CPU-only initialization.
        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = -1;
    } else {
        // GPU initialization on the given device.
        //
        // For CUDA-aware MPI implementations, cudaSetDevice must be called
        // before MPI_Init is called, because MPI_Init will pick up the created
        // CUDA context and use it to create its own internal streams. It uses
        // these internal streams for data transfers, which allows it to
        // implement asynchronous sends and receives and allows it to overlap
        // GPU data transfers with whatever other computation the GPU may be
        // doing.
        //
        // It is not possible to control which streams the MPI implementation
        // uses for its data transfer.
        cudaError_t error = cudaSetDevice(device);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed with an error");
        }

        // When doing a CUDA-aware allreduce, the reduction itself (the
        // summation) must be done on the GPU with an elementwise arithmetic
        // kernel. We create our own stream to launch these kernels on, so that
        // the kernels can run independently of any other computation being done
        // on the GPU.
        cudaStreamCreate(&global_state.stream);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed with an error");
        }

        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = device;
    }
    global_state.initialized = true;
}

// Allocate a new memory buffer on CPU or GPU.
float* alloc(size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        return new float[size];
    } else {
        // GPU memory allocation through CUDA allocator.
        void* memory;
        cudaError_t error = cudaMalloc(&memory, sizeof(float) * size);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed with an error");
        }
        return (float*) memory;
    }
}

// Deallocate an allocated memory buffer.
void dealloc(float* buffer) {
    if(global_state.device < 0) {
        // CPU memory deallocation through standard allocator.
        delete[] buffer;
    } else {
        // GPU memory deallocation through CUDA allocator.
        cudaFree(buffer);
    }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        std::memcpy((void*) dst, (void*) src, size * sizeof(float));
    } else {
        // GPU memory allocation through CUDA allocator.
        cudaMemcpyAsync((void*) dst, (void*) src, size * sizeof(float),
                        cudaMemcpyDeviceToDevice, global_state.stream);
        cudaStreamSynchronize(global_state.stream);
    }
}

// GPU kernel for adding two vectors elementwise.
__global__ void kernel_add(const float* x, const float* y, const int N, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      out[i] = x[i] + y[i];
    }
}


// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size) {
    if(global_state.device < 0) {
        // Accumulate values from `src` into `dst` on the CPU.
        for(size_t i = 0; i < size; i++) {
            dst[i] += src[i];
        }
    } else {
        // Launch a GPU kernel to accumulate values from `src` into `dst`.
        kernel_add<<<32, 256, 0, global_state.stream>>>(src, dst, size, dst);
        cudaStreamSynchronize(global_state.stream);
    }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}

void PrintData(int rank, float* data, size_t length) {
    std::cout << "rank: " << rank << " data: ";
    for(size_t i = 0; i < length; i++) {
        std::cout << data[i] << ",";
    }
    std::cout << '\n';
}


void RecursiveAllreduce(float* data, size_t length, float** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for(size_t other_length : lengths) {
        if(length != other_length) {
            throw std::runtime_error("RingAllreduce received different lengths");
        }
    }

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Allocate the output buffer.
    float* output = alloc(length);
    *output_ptr =  output;

    // Copy your data to the output buffer to avoid modifying the input buffer.
    copy(output, data, length);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    float* buffer = alloc(length / 2);

    int dist = size / 2;
    int cur_size = size;

    // r_vec shows the starting place
    // r_vec ~ r_vec + cur_len shows the range for each node's responsible range for Scatter Reduce
    float* r_vec = output;
    size_t cur_len = length / 2;

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_FLOAT;

    // Scatter Reduce Vector Halving
    while (true) {
        float *segment_send;
        int opponent = (rank + dist) % cur_size + int(rank / cur_size) * cur_size;
        if (opponent < rank) {
            // the later half nodes will send top half of the vector
            segment_send = r_vec;
        } else {
            segment_send = r_vec + cur_len;
        }

        MPI_Send(segment_send, (int)cur_len, datatype, opponent, 0,
                 MPI_COMM_WORLD);
        MPI_Irecv(buffer, (int)cur_len, datatype, opponent, 0, MPI_COMM_WORLD,
                  &recv_req);

        MPI_Wait(&recv_req, &recv_status);
        if (opponent < rank) {
            r_vec += cur_len;
        }
        reduce(r_vec, buffer, cur_len);

        dist /=  2;
        if (dist <= 0) {
            break;
        }
        cur_len /= 2;
        cur_size /= 2;
    }


    dist = 1;
    while (dist <= size / 2) {
        float *segment_send;
        int opponent = (rank + dist) % cur_size + int(rank / cur_size) * cur_size;
        std::cout << "rank :" << rank << "opponent: " << opponent << '\n';
        segment_send = r_vec;

        MPI_Send(segment_send, (int)cur_len, datatype, opponent, 0, MPI_COMM_WORLD);
        MPI_Irecv(buffer, (int)cur_len, datatype, opponent, 0, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&recv_req, &recv_status);

        if (opponent < rank) {
            r_vec -= cur_len;
        } else {
            r_vec += cur_len;
        }
        copy(r_vec, buffer, cur_len);
        if (opponent > rank) {
            r_vec -= cur_len;
        }

        cur_len *= 2;
        dist *= 2;
        cur_size *= 2;
    }

    // Free temporary memory.
    dealloc(buffer);
}
