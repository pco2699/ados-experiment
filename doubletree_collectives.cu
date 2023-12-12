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
        if (root->rank == 0)
        {
            newNode = new Node(size - 1, nullptr, nullptr, nullptr);
        }
        else
        {
            newNode = new Node(root->rank - 1, parent, nullptr, nullptr);
        }

        // Recursively do the same for the left and right subtrees
        newNode->left = createTreeWithLessValue(root->left, newNode, size);
        newNode->right = createTreeWithLessValue(root->right, newNode, size);

        return newNode;
    }
};

void Node::printLevelOrder() const
{
    // Create a queue for level order traversal
    std::queue<const Node *> q;

    // Enqueue root (the calling object represents the root)
    q.push(this);
    std::cout << "Printing level order traversal. Root rank is " << q.front()->rank << "\n";
    while (!q.empty())
    {
        int currQSize = q.size();
        // Print front of queue and remove it from queue
        for (int i = 0; i < currQSize; i++) {
            const Node *current = q.front();
            std::cout<<current->rank<<"("<< current->parent<<")"<<" ";

            q.pop();
            if (current->left != nullptr)
                q.push(current->left);

            // Enqueue right child
            if (current->right != nullptr)
                q.push(current->right);
        }
        std::cout<<"\n";
        // Enqueue left child
    }
    std::cout << "\n";
}

Node *search(struct Node *root, int key)
{
    // Base Cases: root is null or key is present at root
    if (root == NULL)
    {
        return root;
    }

    if (root->rank == key)
    {
        return root;
    }

    // Key is greater than root's key
    if (root->rank < key)
    {
        return search(root->right, key);
    }

    // Key is smaller than root's key
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
    // std::cout<<"Called Collectives\n";
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
    std::vector<Node *> level;

    int diff = 2;

    for (int i = 1; i < size; i = i + diff)
    {
        level.push_back(new Node(i, nullptr, nullptr, nullptr));
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
    }

    Node *root = new Node(0, nullptr, nullptr, level[0]);
    level[0]->parent = root;
    return root;
}

void printData(int rank, float* data, size_t length) {
    std::cout << "rank: " << rank << " data: ";
    for(size_t i = 0; i < length; i++) {
        std::cout << data[i] << ",";
    }
    std::cout << '\n';
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

    // Allocate the output buffer.
    float *output = alloc(length);
    *output_ptr = output;

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    float *buffer1 = alloc(length);
    float *buffer2 = alloc(length);

    Node *firstTree = constructFirstTree(size);
    Node *secondTree = firstTree->createTreeWithLessValue(firstTree, nullptr, size);

    // firstTree->printLevelOrder();
    // secondTree->printLevelOrder();

    Node *nodeInFirstTree;
    Node *nodeInSecondTree;

    if (rank == 0)
    {
        nodeInFirstTree = firstTree;
        nodeInSecondTree = search(secondTree->right, rank);
    }
    else if (rank == size - 1)
    {
        nodeInFirstTree = search(firstTree->right, rank);
        nodeInSecondTree = secondTree;
    }
    else
    {
        nodeInFirstTree = search(firstTree->right, rank);
        nodeInSecondTree = search(secondTree->right, rank);
    }

    MPI_Status recv_status;
    MPI_Status* statuses = (MPI_Status*)malloc(2 * sizeof(MPI_Status));                
    MPI_Request recv_req;
    MPI_Request reqs[2];
    MPI_Datatype datatype = MPI_FLOAT;

    bool recv_flags[2];

    // Reduce Phase

    for (int i=0;i<length;i++) {
        buffer2[i]=0;
        buffer1[i]=0;
    }
    if (rank % 2 == 0) {
        if (nodeInSecondTree->parent != nullptr) {
            MPI_Isend(data, length, MPI_FLOAT, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD, &recv_req);
        }

       if (nodeInFirstTree->left != nullptr) {
            MPI_Irecv(buffer1, length, datatype, nodeInFirstTree->left->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            recv_flags[0]= true;
       }

       if (nodeInFirstTree->right != nullptr) {
            MPI_Irecv(buffer2, length, datatype, nodeInFirstTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
       }
        if (recv_flags[0] && recv_flags[1] ) {
            MPI_Waitall(2, reqs, statuses);
        } else if (recv_flags[0]) {
            MPI_Wait(&reqs[0], &statuses[0]);
        } else {
            MPI_Wait(&reqs[1], &statuses[1]);
        }

       reduce(buffer2, buffer1, length);

        if (nodeInFirstTree->parent != nullptr) {
            MPI_Send(buffer2, length, MPI_FLOAT, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD);
        }
    } else {
        if (nodeInFirstTree->parent != nullptr) {
            MPI_Isend(data, length, MPI_FLOAT, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD, &recv_req);
        }

       if (nodeInSecondTree->left != nullptr) {
            MPI_Irecv(buffer1, length, datatype, nodeInSecondTree->left->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            recv_flags[0]=true;
       }

       if (nodeInSecondTree->right != nullptr) {
            MPI_Irecv(buffer2, length, datatype, nodeInSecondTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
       }

        if (recv_flags[0] && recv_flags[1] ) {
            MPI_Waitall(2, reqs, statuses);
        } else if (recv_flags[0]) {
            MPI_Wait(&reqs[0], &statuses[0]);
        } else {
            MPI_Wait(&reqs[1], &statuses[1]);
        }

       reduce(buffer2, buffer1, length);

        if (nodeInSecondTree->parent != nullptr) {
            MPI_Send(buffer2, length, MPI_FLOAT, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD);
        }
    }

    // Wait for Send to opposite tree
    MPI_Wait(&recv_req, &recv_status);

    // Reduce phase is complete
    // Start broadcast

    for (int i = 0; i<length; i++) {
        output[i]=0;
    }

    reqs[0] = MPI_REQUEST_NULL;
    reqs[1] = MPI_REQUEST_NULL;
    recv_flags[0] = false;
    recv_flags[1] = false;

    if (rank % 2 == 0) {
        if (nodeInFirstTree->parent != nullptr) {
            MPI_Recv(buffer1, length, datatype, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            reduce(output, buffer1, length);
        } else {
            reduce(output, buffer2, length);
        }

        if (nodeInFirstTree->left != nullptr) {
            MPI_Isend(output, length, MPI_FLOAT, nodeInFirstTree->left->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            recv_flags[0]=true;
        }
        if (nodeInFirstTree->right != nullptr && nodeInFirstTree->parent != nullptr) {
            MPI_Isend(output, length, MPI_FLOAT, nodeInFirstTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
        } else if (nodeInFirstTree->right != nullptr && nodeInFirstTree->parent == nullptr) {
            MPI_Isend(buffer2, length, MPI_FLOAT, nodeInFirstTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
        }

        if (nodeInSecondTree->parent != nullptr) {
            MPI_Recv(buffer1, length, datatype, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        reduce(output, buffer1, length);
    } else {
        if (nodeInSecondTree->parent != nullptr) {
            MPI_Recv(buffer1, length, datatype, nodeInSecondTree->parent->rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            reduce(output, buffer1, length);
        } else {
            reduce(output, buffer2, length);
        }

        if (nodeInSecondTree->left != nullptr) {
            MPI_Isend(output, length, MPI_FLOAT, nodeInSecondTree->left->rank, 0, MPI_COMM_WORLD, &reqs[0]);
            recv_flags[0]=true;
        }
        if (nodeInSecondTree->right != nullptr && nodeInSecondTree->parent != nullptr) {
            MPI_Isend(output, length, MPI_FLOAT, nodeInSecondTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
        } else if(nodeInSecondTree->right != nullptr && nodeInSecondTree->parent == nullptr) {
            MPI_Isend(buffer2, length, MPI_FLOAT, nodeInSecondTree->right->rank, 0, MPI_COMM_WORLD, &reqs[1]);
            recv_flags[1]=true;
        }

        if (nodeInFirstTree->parent != nullptr) {
            MPI_Recv(buffer1, length, datatype, nodeInFirstTree->parent->rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        reduce(output, buffer1, length);
    }

    if (recv_flags[0] && recv_flags[1] ) {
        MPI_Waitall(2, reqs, statuses);
    } else if (recv_flags[0]) {
        MPI_Wait(&reqs[0], &statuses[0]);
    } else {
        MPI_Wait(&reqs[1], &statuses[1]);
    }


    // Reduce phase is complete
    // Start broadcast
    dealloc(buffer1);
    dealloc(buffer2);
}
/*
    If current node is odd (Need to recv from one odd node(if i am not root), and one even node)
        If target nodes are even:
            Recv(from odd); Wait(); Reduce(); Send(all except root);      Recv(from even); Send(root); Wait(); Reduce()
        else:
            Send(); Recv(all); Wait(all); Reduce();

    If current node is even
        If target nodes are odd:
            if evenRecv is root:
                Send(all?); RecvAll(); Wait(); Reduce() 
            else:
                Recv(from even); Wait(); Reduce(); Send(all except root); Recv(from odd); Send(root); Wait(); Reduce()
        else:
            Send(); Recv(all); Wait(all); Reduce();

            0(1) sendTo(2, 1); IRecv(1) - Waiting for 1
            1(1) IRecv(3) - Waiting for 3
            2(2) IRecv(0) - succeed - Reduce - SendTo(1, 2); Send(3, 2); IRecv(1) - Waiting for 1
            3(3) Send(1, 1); IRecv(2) - Succeed - Reduce

            0(1+4) SendTo(2, 1); Recv(1) - Reduce() - Done
            1(4) Recv(3) - Reduce(from 3);  SendTo(2, 2); Recv(2) ; Reduce(); SendTo(0, 4) - Done
            2(2) Recv(0) ; Reduce(); SendTo(1,2); Recv(0) - 
            3(1) Send(1, 1); Recv(2) - Waiting for 2

            0(1) Recv(1) - Waiting for 1
            1(1) Recv(3) - Waiting for 3
            2(1) Recv(0) - Waiting for 0
            3(1) Recv()



            0
            |
             2-1
            / \
            1  3


3->1, 2(after reducing even)->1
0->2, 1(after reducing odd)->2
1(after reduce)->0
2(after reduce)->3
*/