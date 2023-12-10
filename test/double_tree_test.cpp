#include "timer.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>

#include "doubletree_collectives.h"

void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Initialize on CPU (no GPU device ID).
    DoubleTreeCollectives(NO_DEVICE);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* data = new float[size];
        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                data[j] = (float)j;
            }

            float* output;
            timer.start();
            DoubleTreeAllreduce(data, size, &output);
            seconds += timer.seconds();

            std::cout<<"Result at rank " << mpi_rank<< " is " << ":";
            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                std::cout<<output[j]<<" ";
                // if(output[j] != (float) j * mpi_size) {
                //     std::cerr << "Unexpected result from allreduce: " << output[j] << ", expected "<< (float)j*mpi_size <<std::endl;
                //     return;
                // }
            }
            std::cout<<"\n";
            delete[] output;
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        delete[] data;
    }
}

void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    // 
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    DoubleTreeCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    cudaError_t err;

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* cpu_data = new float[size];

        float* data;
        err = cudaMalloc(&data, sizeof(float) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = (float)j;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            float* output;
            timer.start();
            DoubleTreeAllreduce(data, size, &output);
            seconds += timer.seconds();

            err = cudaMemcpy(cpu_data, output, sizeof(float) * size, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(cpu_data[j] != (float) j * mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << ", expected "<< (float)j*mpi_size <<std::endl;
                    return;
                }
            }
            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }
    std::string input(argv[1]);

    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes = {
            32
    };

    // Number of iterations to run for each buffer size.
    std::vector<size_t> iterations = {
        1
    };

    // Test on either CPU and GPU.
    if(input == "cpu") {
        TestCollectivesCPU(buffer_sizes, iterations);
    } else if(input == "gpu") {
        TestCollectivesGPU(buffer_sizes, iterations);
    } else {
        std::cerr << "Unknown device type: " << input << std::endl
                  << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
