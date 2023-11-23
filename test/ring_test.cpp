#include <mpi.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>

void TestRingAllreduce() {
    MPI_Init(NULL, NULL);

    // Simulate MPI environment
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generate mock input data (representing gradients or neural network parameters)
    size_t length = 1000000; // Adjust this size based on your needs
    std::vector<float> data(length);

    // Fill the vector with random data
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (auto& value : data) {
        value = distribution(generator);
    }

    float* output_ptr = nullptr;

    // Measure latency
    auto start = std::chrono::high_resolution_clock::now();

    // Call the RingAllreduce function
    RingAllreduce(data.data(), data.size(), &output_ptr);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> latency = end - start;
    std::cout << "Node " << rank << " - All-reduce latency: " << latency.count() << " ms" << std::endl;

    // Cleanup
    delete[] output_ptr;
    MPI_Finalize();
}

int main() {
    TestRingAllreduce();
    return 0;
}
