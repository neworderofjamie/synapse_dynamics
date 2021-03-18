// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define SEED 124

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }


template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
template<typename A = std::milli>
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

enum class Model
{
    Continuous,
    eProp,
};

__global__ void continuousDense(unsigned int numPre, unsigned int numPost, 
                                float *d_output, float *d_ePre, float *d_g)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < (numPre * numPost)) {
        atomicAdd(&d_output[id % numPost], d_g[id] * d_ePre[id / numPost]);
    }
}

__global__ void continuousDenseFastDivide(unsigned int numSynapse, unsigned int numPost, uint32_t numPostA, uint32_t numPostB, uint32_t numPostM,
                                          float *d_output, float *d_ePre, float *d_g)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numSynapse) {
        const unsigned int pre = (((uint64_t)id * numPostA) + numPostB) >> (32 + numPostM);
        const unsigned int post = id - (numPost * pre);
        atomicAdd(&d_output[post], d_g[id] * d_ePre[pre]);
    }
}

__global__ void continuousDenseSharedPre(unsigned int numPre, unsigned int numPost, unsigned int numPostPadded,
                                         float *d_output, float *d_ePre, float *d_g)
{
    __shared__ float s_ePre;

    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int idPre = id / numPostPadded;
    const unsigned int idPost = id % numPostPadded;
    const unsigned int idSyn = (idPre * numPost) + idPost;

    if(threadIdx.x == 0) {
        s_ePre = d_ePre[idPre];
    }
    __syncthreads();
    if(idPost < numPost) {
        atomicAdd(&d_output[idPost], d_g[idSyn] * s_ePre);
    }
}

__global__ void continuousDenseNPre(unsigned int numPre, unsigned int numPost, unsigned int numPostPadded,
                                    float *d_output, float *d_ePre, float *d_g)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int idPre = id / numPostPadded;
    const unsigned int idPost = id % numPostPadded;
    unsigned int idSyn = (idPre * numPost) + idPost;

    if(idPost < numPost) {
        float output = 0.0f;
        for(unsigned int i = 0; i < N; i++) {
            output += d_g[idSyn] * d_ePre[idPre + i];

            idSyn += numPost;
        }
        atomicAdd(&d_output[idPost], output);
    }
}

__global__ void continuousDenseThreadPerPost(unsigned int numPre, unsigned int numPost, 
                                             float *d_output, float *d_ePre, float *d_g)
{
    extern __shared__ float s_ePre[];

    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int numBlocks = (numPre + blockDim.x - 1) / blockDim.x;

    float output = 0.0f;
    for(unsigned int b = 0; b < numBlocks; b++) {
        // Determine how many presynaptic neurons are in this block
        const unsigned int numPreInBlock = (b == (numBlocks - 1))
            ? ((numPre - 1) % blockDim.x) + 1 : blockDim.x;

        __syncthreads();

        // Use first threads in block to ePre into shared memory
        if(threadIdx.x < numPreInBlock) {
            s_ePre[threadIdx.x] = d_ePre[(b * blockDim.x) + threadIdx.x];
        }

        __syncthreads();

        if(id < numPost) {
            unsigned int synAddress = (b * blockDim.x * numPost) + id;
            for(unsigned int i = 0; i < numPreInBlock; i++, synAddress += numPost) {
                output += d_g[synAddress] * d_ePre[i];
            }
        }
    }

    d_output[id] += output;
}

//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(size_t count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, size_t count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, size_t count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
std::tuple<uint32_t, uint32_t, uint32_t> calcFastDivideConstants(uint32_t d)
{
    const uint32_t m = (uint32_t)std::floor(std::log2(d));

    const uint32_t uintMax = std::numeric_limits<uint32_t>::max();
    if(d == (1 << m)) {
        return std::make_tuple(uintMax, uintMax, m);
    }
    else {
        const uint32_t t = (1ull << (m + 32)) / d;
        const uint32_t r = ((t * d) + d) & uintMax;
        if(r <= (1u << m)) {
            return std::make_tuple(t + 1, 0, m);
        }
        else {
            return std::make_tuple(t, t, m);
        }
    }
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    unsigned int blockSize = 32;
    unsigned int numNeurons = 10000;

    // Read mode from command line
    Model model;
    if(argc < 2) {
        std::cerr << "Expected parameters specifying:" << std::endl;
        std::cerr << "\t Model (0 = Continuous, 1 = eProp)" << std::endl;
        return EXIT_FAILURE;
    }
    else {
        model = static_cast<Model>(std::stoul(argv[1]));
    }

    // If additional parameters are specified, read N
    if(argc > 2) {
        numNeurons = std::stoul(argv[2]);
    }
    if(argc > 3) {
        blockSize = std::stoul(argv[3]);
    }

    std::cout << "Model:" << static_cast<int>(model) << ", num neurons:" << numNeurons << ", block size:" << blockSize << std::endl;

    // Calculate fast divide constants
    uint32_t numPostA;
    uint32_t numPostB;
    uint32_t numPostM;
    std::tie(numPostA, numPostB, numPostM) = calcFastDivideConstants(numNeurons);

    const unsigned int numSynapses = numNeurons * numNeurons;

    const unsigned int numNeuronBlocks = (numNeurons + blockSize - 1) / blockSize;

    CHECK_CUDA_ERRORS(cudaSetDevice(0));

    std::mt19937 rng;
    std::normal_distribution<float> dis(0.0f, 1.0f);

    if(model == Model::Continuous) {
        // Create array to hold post-synaptic output
        auto output = allocateHostDevice<float>(numNeurons);
        
        // Create arrays to hold presynaptic state and weight
        auto ePre = allocateHostDevice<float>(numNeurons);
        auto g = allocateHostDevice<float>(numSynapses);

        // Randomize
        std::generate_n(&ePre.first[0], numNeurons, [&rng, &dis]() { return dis(rng); });
        std::generate_n(&g.first[0], numSynapses, [&rng, &dis]() { return dis(rng); });

        hostToDeviceCopy(ePre, numNeurons);
        hostToDeviceCopy(g, numSynapses);

        // Zero output
        std::fill_n(&output.first[0], numNeurons, 0.0f);
        hostToDeviceCopy(output, numNeurons);

        {
            Timer<std::milli> t("Continuous Dense:");
            const unsigned int numBlocks = (numSynapses + blockSize - 1) / blockSize;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            for(unsigned int i = 0; i < 5000; i++) {
                continuousDense<<<grid, threads>>>(numNeurons, numNeurons, output.second, ePre.second, g.second);
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }

        // Zero output
        std::fill_n(&output.first[0], numNeurons, 0.0f);
        hostToDeviceCopy(output, numNeurons);

        {
            Timer<std::milli> t("Continuous Dense Fast Divide:");
            const unsigned int numBlocks = (numSynapses + blockSize - 1) / blockSize;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            for(unsigned int i = 0; i < 5000; i++) {
                continuousDenseFastDivide<<<grid, threads>>>(numSynapses, numNeurons, numPostA, numPostB, numPostM,
                                                             output.second, ePre.second, g.second);
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }

        // Zero output
        std::fill_n(&output.first[0], numNeurons, 0.0f);
        hostToDeviceCopy(output, numNeurons);

        {
            Timer<std::milli> t("Continuous Dense Shared Pre:");
            const unsigned int numBlocks = numNeuronBlocks * numNeurons;
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            for(unsigned int i = 0; i < 5000; i++) {
                continuousDenseSharedPre<<<grid, threads>>>(numNeurons, numNeurons, numNeuronBlocks * blockSize,
                                                            output.second, ePre.second, g.second);
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }

        // Zero output
        /*std::fill_n(&output.first[0], numNeurons, 0.0f);
        hostToDeviceCopy(output, numNeurons);

        {
            Timer<std::milli> t("Continuous Dense Thread Per Post:");
            const unsigned int numBlocks = (numSynapses + blockSize - 1) / blockSize;
            const unsigned int sharedBytes = blockSize * sizeof(unsigned int);
            dim3 threads(blockSize, 1);
            dim3 grid(numBlocks, 1);

            for(unsigned int i = 0; i < 5000; i++) {
                continuousDenseThreadPerPost<<<grid, threads, sharedBytes>>>(numNeurons, numNeurons, output.second, ePre.second, g.second);
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }*/
    }
    return EXIT_SUCCESS;
}