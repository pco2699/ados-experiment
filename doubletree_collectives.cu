#include <vector>
#include <queue>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>

#include "doubletree_collectives.h"

struct MPIGlobalState
{
    // The CUDA device to run on, or -1 for CPU-only.
    int device = -1;

    // A CUDA stream (if device >= 0) initialized on the device
    cudaStream_t stream;

    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
};

class Node
{
public:
    int rank;
    Node *parent;
    Node *left;
    Node *right;

    Node(int rank, Node *p, Node *l, Node *r)
        : rank(rank), parent(p), left(l), right(r) {}

    void printLevelOrder() const;

    static Node *createTreeWithLessValue(Node *root, Node *parent, int size)
    {
        if (root == nullptr)
        {
            return nullptr;
        }

        Node *newNode;
        if (root-> rank == 0) {
            newNode = new Node(size-1, nullptr, nullptr, nullptr);
        } else {
            newNode = new Node(root->rank - 1, parent, nullptr, nullptr);
        }

        // Recursively do the same for the left and right subtrees
        newNode->left = createTreeWithLessValue(root->left, parent, size);
        newNode->right = createTreeWithLessValue(root->right, parent, size);

        return newNode;
    }
};

void Node::printLevelOrder() const
{
    // Create a queue for level order traversal
    std::queue<const Node *> q;

    // Enqueue root (the calling object represents the root)
    q.push(this);
    std::cout << "Printing level order traversal. Root rank is "<< q.front()->rank << "\n";
    while (!q.empty())
    {
        // Print front of queue and remove it from queue
        const Node *current = q.front();
        std::cout<<current->rank<<" ";
        q.pop();

        // Enqueue left child
        if (current->left != nullptr)
            q.push(current->left);

        // Enqueue right child
        if (current->right != nullptr)
            q.push(current->right);
    }
    std::cout<<"\n";
}

Node *search(struct Node *root, int key)
{
    // std::cout<<"Calling search with key="<<key<<"and root->rank="<<root->rank<<"\n";
    // Base Cases: root is null or key is present at root
    if (root == NULL) {
        // std::cout<<"Not Found:"<<key<<"\n";
        return root;
    }
    
    if (root->rank == key) {
        // std::cout<<"Found here\n";    
        return root;
    }

    // Key is greater than root's key
    if (root->rank < key) {
        // std::cout <<"Moving right\n";
        return search(root->right, key);
    }

    // Key is smaller than root's key
    // std::cout <<"Moving left\n";
    return search(root->left, key);
}
// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void DoubleTreeCollectives(int device)
{
    if (device < 0)
    {
        // CPU-only initialization.
        int mpi_error = MPI_Init(NULL, NULL);
        if (mpi_error != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = -1;
    }
    else
    {
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
        if (error != cudaSuccess)
        {
            throw std::runtime_error("cudaSetDevice failed with an error");
        }

        // When doing a CUDA-aware allreduce, the reduction itself (the
        // summation) must be done on the GPU with an elementwise arithmetic
        // kernel. We create our own stream to launch these kernels on, so that
        // the kernels can run independently of any other computation being done
        // on the GPU.
        cudaStreamCreate(&global_state.stream);
        if (error != cudaSuccess)
        {
            throw std::runtime_error("cudaStreamCreate failed with an error");
        }

        int mpi_error = MPI_Init(NULL, NULL);
        if (mpi_error != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = device;
    }
    global_state.initialized = true;
}

// Allocate a new memory buffer on CPU or GPU.
float *alloc(size_t size)
{
    if (global_state.device < 0)
    {
        // CPU memory allocation through standard allocator.
        return new float[size];
    }
    else
    {
        // GPU memory allocation through CUDA allocator.
        void *memory;
        cudaError_t error = cudaMalloc(&memory, sizeof(float) * size);
        if (error != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed with an error");
        }
        return (float *)memory;
    }
}

// Deallocate an allocated memory buffer.
void dealloc(float *buffer)
{
    if (global_state.device < 0)
    {
        // CPU memory deallocation through standard allocator.
        delete[] buffer;
    }
    else
    {
        // GPU memory deallocation through CUDA allocator.
        cudaFree(buffer);
    }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float *dst, float *src, size_t size)
{
    if (global_state.device < 0)
    {
        // CPU memory allocation through standard allocator.
        std::memcpy((void *)dst, (void *)src, size * sizeof(float));
    }
    else
    {
        // GPU memory allocation through CUDA allocator.
        cudaMemcpyAsync((void *)dst, (void *)src, size * sizeof(float),
                        cudaMemcpyDeviceToDevice, global_state.stream);
        cudaStreamSynchronize(global_state.stream);
    }
}

// GPU kernel for adding two vectors elementwise.
__global__ void kernel_add(const float *x, const float *y, const int N, float *out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        out[i] = x[i] + y[i];
    }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float *dst, float *src, size_t size)
{
    if (global_state.device < 0)
    {
        // Accumulate values from `src` into `dst` on the CPU.
        for (size_t i = 0; i < size; i++)
        {
            dst[i] += src[i];
        }
    }
    else
    {
        // Launch a GPU kernel to accumulate values from `src` into `dst`.
        kernel_add<<<32, 256, 0, global_state.stream>>>(src, dst, size, dst);
        cudaStreamSynchronize(global_state.stream);
    }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length)
{
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}

// size must be a power of 2
Node *constructFirstTree(int size)
{
    // Node* leafs = new arr

    // std::cout << "Constructing tree with size="<<size<<"\n";
    std::vector<Node *> level;

    int diff = 2;

    for (int i = 1; i < size; i = i + diff)
    {
        level.push_back(new Node(i, nullptr, nullptr, nullptr));
    }
    // std::cout << level.size();

    for (const auto& node: level) {
        std::cout << node-> rank << " ";
    }
    diff = diff * 2;

    while (diff <= size)
    {

        std::vector<Node *> prevLevel = level;
        level.clear();

        for (int i = 0; i < prevLevel.size() - 1; i += 2)
        {
            Node *node = new Node(prevLevel[i]->rank + diff / 4, nullptr, prevLevel[i], prevLevel[i + 1]);
            level.push_back(node);
            prevLevel[i]->parent = node;
            prevLevel[i + 1]->parent = node;
        }

        diff = diff * 2;
        // for (const auto& node: level) {
        //     std::cout << node-> rank << " ";
        // }
        // std::cout << "\n";
    }

    Node* root = new Node(0, nullptr, nullptr, level[0]);
    level[0]-> parent = root;
    return root;
}

void DoubleTreeAllreduce(float *data, size_t length, float **output_ptr)
{
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for (size_t other_length : lengths)
    {
        if (length != other_length)
        {
            throw std::runtime_error("DoubleTreeAllreduce received different lengths");
        }
    }

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i)
    {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i)
    {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == length);

    // Allocate the output buffer.
    float *output = alloc(length);
    *output_ptr = output;

    // Copy your data to the output buffer to avoid modifying the input buffer.
    copy(output, data, length);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    float *buffer = alloc(segment_sizes[0]);

    Node *firstTree = constructFirstTree(size);
    Node *secondTree = firstTree->createTreeWithLessValue(firstTree, nullptr, size);

    // firstTree->printLevelOrder();

    Node *nodeInFirstTree;
    Node *nodeInSecondTree;

    if (rank == 0) {
        nodeInFirstTree = firstTree;
        // std::cout << "Searching 2nd tree\n";
        nodeInSecondTree = search(secondTree->right, rank);
    } else if (rank == size - 1) {
        // std::cout << "Searching 1st tree\n";
        nodeInFirstTree = search(firstTree->right, rank);
        nodeInSecondTree = secondTree;
    } else {
        // std::cout << "Searching 1st tree\n";
        nodeInFirstTree = search(firstTree->right, rank);
        // std::cout << "Searching 2nd tree\n";
        nodeInSecondTree = search(secondTree->right, rank);
    }

    // std::cout << "Searched rank="<< rank << ". Found:" <<nodeInFirstTree->rank << "\n";
    // std::cout << "Searched rank="<< rank << ". Found:" <<nodeInSecondTree->rank << "\n";
    
    // MPI_Status recv_status1;
    // MPI_Status recv_status2;
    // MPI_Request recv_req1;
    // MPI_Request recv_req2;
    MPI_Datatype datatype = MPI_FLOAT;

    MPI_Request reqs[2]; //= {recv_req1, recv_req2};
    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).

    std::cout << "Node Rank is "<<rank<<"\n";

    for (int i = 0; i < size - 1; i++)
    {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        float *segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);
        if (rank%2 == 0) {
            if (nodeInFirstTree->parent != nullptr) {
                std::cout<<"Receiving at "<<rank<<" from "<<nodeInFirstTree->parent->rank<<"\n";
                MPI_Irecv(buffer, segment_sizes[recv_chunk],
                        datatype, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            }

            if (nodeInSecondTree->parent != nullptr) {
                std::cout<<"Receiving at "<<rank<<" from "<<nodeInSecondTree->parent->rank<<"\n";
                MPI_Irecv(buffer, segment_sizes[recv_chunk],
                        datatype, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            }

            if (nodeInFirstTree->left != nullptr) {
                
                MPI_Send(segment_send, segment_sizes[send_chunk],MPI_FLOAT, nodeInFirstTree->left->rank, 0, MPI_COMM_WORLD);
            }

            if (nodeInFirstTree->right != nullptr)
                MPI_Send(segment_send, segment_sizes[send_chunk],MPI_FLOAT, nodeInFirstTree->right->rank, 0, MPI_COMM_WORLD);
        } else {
            if (nodeInSecondTree->parent != nullptr) {
                std::cout<<"Receiving at "<<rank<<" from "<<nodeInSecondTree->parent->rank<<"\n";
                MPI_Irecv(buffer, segment_sizes[recv_chunk],
                        datatype, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            }

            if (nodeInFirstTree->parent != nullptr)
                MPI_Irecv(buffer, segment_sizes[recv_chunk],
                        datatype, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD, &reqs[1]);

            if (nodeInSecondTree->left != nullptr)
                MPI_Send(segment_send, segment_sizes[send_chunk],MPI_FLOAT, nodeInSecondTree->left->rank, 0, MPI_COMM_WORLD);

            if (nodeInSecondTree->right != nullptr)
                MPI_Send(segment_send, segment_sizes[send_chunk],MPI_FLOAT, nodeInSecondTree->right->rank, 0, MPI_COMM_WORLD);
        }

        float *segment_update = &(output[segment_ends[recv_chunk] -
                                         segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        std::cout<<"Waiting for recv\n";
        // MPI_Request reqs[2] = {recv_req1, recv_req2};
        assert(reqs != nullptr);
        for (int i = 0; i < 2; i++) {
            assert(reqs[i] != MPI_REQUEST_NULL);
        }
        MPI_Status statuses[2];

        assert(statuses != nullptr);
        MPI_Waitall(2, reqs, statuses);
        // MPI_Wait(&recv_req1, &recv_status1);
        // std::cout<<"One received\n";
        // MPI_Wait(&recv_req2, &recv_status2);

        std::cout<<"Reducing\n";
        reduce(segment_update, buffer, segment_sizes[recv_chunk]);
        std::cout<<"Reducing complete\n";
    }

    std::cout << "Finished for node with rank "<<rank<<"\n";
    // Free temporary memory.
    dealloc(buffer);
}

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
//
// For more information on the ring allgather, read the documentation for the
// ring allreduce, which includes a ring allgather as the second stage.
// void RingAllgather(float *data, size_t length, float **output_ptr)
// {
//     // Get MPI size and rank.
//     int rank;
//     int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     if (mpi_error != MPI_SUCCESS)
//         throw std::runtime_error("MPI_Comm_rank failed with an error");

//     int size;
//     mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
//     if (mpi_error != MPI_SUCCESS)
//         throw std::runtime_error("MPI_Comm_size failed with an error");

//     // Get the lengths of data provided to every process, so that we know how
//     // much memory to allocate for the output buffer.
//     std::vector<size_t> segment_sizes = AllgatherInputLengths(size, length);
//     size_t total_length = 0;
//     for (size_t other_length : segment_sizes)
//     {
//         total_length += other_length;
//     }

//     // Compute where each chunk ends.
//     std::vector<size_t> segment_ends(size);
//     segment_ends[0] = segment_sizes[0];
//     for (size_t i = 1; i < segment_ends.size(); ++i)
//     {
//         segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
//     }

//     assert(segment_sizes[rank] == length);
//     assert(segment_ends[size - 1] == total_length);

//     // Allocate the output buffer and copy the input buffer to the right place
//     // in the output buffer.
//     float *output = alloc(total_length);
//     *output_ptr = output;

//     copy(output + segment_ends[rank] - segment_sizes[rank],
//          data, segment_sizes[rank]);

//     // Receive from your left neighbor with wrap-around.
//     const size_t recv_from = (rank - 1 + size) % size;

//     // Send to your right neighbor with wrap-around.
//     const size_t send_to = (rank + 1) % size;

//     // What type of data is being sent
//     MPI_Datatype datatype = MPI_FLOAT;

//     MPI_Status recv_status;

//     // Now start pipelined ring allgather. At every step, for every rank, we
//     // iterate through segments with wraparound and send and recv from our
//     // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
//     // and receives segment (rank - i).
//     for (size_t i = 0; i < size_t(size - 1); ++i)
//     {
//         int send_chunk = (rank - i + size) % size;
//         int recv_chunk = (rank - i - 1 + size) % size;
//         // Segment to send - at every iteration we send segment (r+1-i)
//         float *segment_send = &(output[segment_ends[send_chunk] -
//                                        segment_sizes[send_chunk]]);

//         // Segment to recv - at every iteration we receive segment (r-i)
//         float *segment_recv = &(output[segment_ends[recv_chunk] -
//                                        segment_sizes[recv_chunk]]);
//         MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
//                      datatype, send_to, 0, segment_recv,
//                      segment_sizes[recv_chunk], datatype, recv_from,
//                      0, MPI_COMM_WORLD, &recv_status);
//     }
// }
