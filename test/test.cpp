#include "collectives.h"
#include "timer.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <iostream>

// Memory checking function
void checkGPUMemory(const std::string& stage) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status){
        std::cerr << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "GPU memory usage at " << stage << ": used = " << used_db / 1024.0 / 1024.0 
              << " MB, free = " << free_db / 1024.0 / 1024.0 
              << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}

void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Initialize on CPU (no GPU device ID).
    InitCollectives(NO_DEVICE);

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
        //float seconds = 0.0f;
        float total_seconds = 0.0f;
        float total_bandwidth = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                data[j] = 1.0f;
            }

            float* output;
            timer.start();
            RingAllreduce(data, size, &output);
            //seconds += timer.seconds();
            float iteration_time = timer.seconds(); // Measure the time for this iteration
            total_seconds += iteration_time; // Add to total time

            // Bandwidth calculation
            size_t total_data_transferred = 2 * size * sizeof(int8_t); // total data in bytes
            float bandwidth = total_data_transferred / iteration_time; // bandwidth in bytes per second
            total_bandwidth += bandwidth;

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(output[j] != (float) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << data[j] << std::endl;
                    return;
                }
            }
            delete[] output;
        }
        if(mpi_rank == 0) {
            float average_time = total_seconds / iters;
            float average_bandwidth = total_bandwidth / iters;
            std::cout << "Verified Ring Allreduce for size " << size
                      << " (Average time: " << average_time << " seconds per iteration, "
                      << "Average Bandwidth: " << average_bandwidth << " Bytes/second)" << std::endl;
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

    // Check and print GPU memory usage before the operation

    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    //int gpu_to_use = mpi_local_rank % num_gpus;
    InitCollectives(mpi_local_rank);

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

        //checkGPUMemory("Before operation for size " + std::to_string(size));

        //Added for bandwidth
        float total_seconds = 0.0f;
        float total_bandwidth = 0.0f;

        float* cpu_data = new float[size];

        float* data;
        err = cudaMalloc(&data, sizeof(float) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = 1.0f;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            float* output;
            timer.start();
            RingAllreduce(data, size, &output);
            float iteration_time = timer.seconds();
            seconds += iteration_time;
            total_seconds += iteration_time;

            // Bandwidth calculation
            size_t total_data_transferred = 2 * size * sizeof(float); // total data in bytes
            float bandwidth = total_data_transferred / seconds; // bandwidth in bytes per second
            total_bandwidth += bandwidth;            

            err = cudaMemcpy(cpu_data, output, sizeof(float) * size, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(cpu_data[j] != (float) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
                    return;
                }
            }
            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }
        if(mpi_rank == 0) {

            float average_time = total_seconds / iters;
            float average_bandwidth = total_bandwidth / iters;
            std::cout << "Verified Ring allreduce for size " << size
                      << " (Average time: " << average_time << " seconds per iteration, "
                      << "Average Bandwidth: " << average_bandwidth << " Bytes/second)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
        //checkGPUMemory("After operation for size " + std::to_string(size));
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: allreduce-ring-test (cpu|gpu)" << std::endl;
        return 1;
    }
    std::string input(argv[1]);

    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes = {
        0, 2, 4, 8, 16, 32, 64, 128, 256
    };

    // Number of iterations to run for each buffer size.
    std::vector<size_t> iterations = {
        100000, 100000, 100000, 100000,
        1000, 1000, 1000, 1000,
        100
    };

    //int num_gpus = 2;

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
