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
                                float *d_output, const float *d_ePre, const float *d_g)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < (numPre * numPost)) {
        atomicAdd(&d_output[id % numPost], d_g[id] * d_ePre[id / numPost]);
    }
}

__global__ void epropALIFDense(unsigned int numPre, unsigned int numPost,
                               const float *d_zFilterPre,  const float *d_psiPost,
                               const float *d_fAvgPost, const float *d_ePost,
                               float * d_eFiltered, float *d_epsilonA, float *d_deltaG)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    constexpr float alpha = 0.95; // exp(-1.0f/20.0f)
    constexpr float rho = 0.999f; // exp(-1.0f/2000.0f)
    constexpr float fTargetTimestep = 0.01f; // (10.0f * 1.0f) / 1000.0f

    constexpr float cReg = 1.0f / 500000.0f;
    constexpr float beta = 0.0174f;

    if(id < (numPre * numPost)) {
        const unsigned int idPre = id / numPost;
        const unsigned int idPost = id % numPost;

        // Calculate some common factors in e and epsilon update
        float epsilonA = d_epsilonA[id];
        const float psiZFilter = d_psiPost[idPost] * d_zFilterPre[idPre];
        const float psiBetaEpsilonA = d_psiPost[idPost] * beta * epsilonA;

        // Calculate e and episilonA
        const float e = psiZFilter - psiBetaEpsilonA;
        d_epsilonA[id] = psiZFilter + ((rho * epsilonA) - psiBetaEpsilonA);

        // Calculate filtered version of eligibility trace
        float eFiltered = d_eFiltered[id];
        eFiltered = (eFiltered * alpha) + e;

        // Apply weight update
        d_deltaG[id] += (eFiltered * d_ePost[idPost]) + ((d_fAvgPost[idPost] - fTargetTimestep) * cReg * e);
        d_eFiltered[id] = eFiltered;
    }
}

__global__ void continuousDenseBlock(unsigned int numPre, unsigned int numPost, unsigned int numPostPadded,
                                     float *d_output, const float *d_ePre, const float *d_g)
{
    extern __shared__ float s_ePre[];

    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int idPost = id % numPostPadded;

    // Get indices of first presynaptic neuron and synapse
    const unsigned int idPreStart = (id / numPostPadded) * blockDim.x;
    const unsigned int idPreEnd = min(idPreStart + blockDim.x, numPre);
    const unsigned int numPreInBlock = idPreEnd - idPreStart;
    unsigned int idSyn = (idPreStart * numPost) + idPost;

    // Use first threads in block to ePre into shared memory
    if(threadIdx.x < numPreInBlock) {
        s_ePre[threadIdx.x] = d_ePre[idPreStart + threadIdx.x];
    }

    __syncthreads();

    if(idPost < numPost) {
        float output = 0.0f;
        for(unsigned int i = idPreStart; i < idPreEnd; i++) {
            output += d_g[idSyn] * s_ePre[i - idPreStart];

            idSyn += numPost;
        }
        atomicAdd(&d_output[idPost], output);
    }
}


__global__ void epropALIFDenseBlock(unsigned int numPre, unsigned int numPost, unsigned int numPostPadded,
                                    const float *d_zFilterPre, const float *d_psiPost,
                                    const float *d_fAvgPost, const float *d_ePost,
                                    float *d_eFiltered, float *d_epsilonA, float *d_deltaG)
{
    extern __shared__ float s_zFilterPre[];

    constexpr float alpha = 0.95; // exp(-1.0f/20.0f)
    constexpr float rho = 0.999f; // exp(-1.0f/2000.0f)
    constexpr float fTargetTimestep = 0.01f; // (10.0f * 1.0f) / 1000.0f

    constexpr float cReg = 1.0f / 500000.0f;
    constexpr float beta = 0.0174f;

    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);
    const unsigned int idPost = id % numPostPadded;

    // Get indices of first presynaptic neuron and synapse
    const unsigned int idPreStart = (id / numPostPadded) * blockDim.x;
    const unsigned int idPreEnd = min(idPreStart + blockDim.x, numPre);
    const unsigned int numPreInBlock = idPreEnd - idPreStart;
    unsigned int idSyn = (idPreStart * numPost) + idPost;

    // Use first threads in block to ePre into shared memory
    if(threadIdx.x < numPreInBlock) {
        s_zFilterPre[threadIdx.x] = d_zFilterPre[idPreStart + threadIdx.x];
    }

    __syncthreads();

    if(idPost < numPost) {
        const float psiPost = d_psiPost[idPost];
        const float fAvgPost = d_fAvgPost[idPost];
        const float ePost = d_ePost[idPost];

        for(unsigned int i = idPreStart; i < idPreEnd; i++) {
            // Calculate some common factors in e and epsilon update
            float epsilonA = d_epsilonA[idSyn];
            const float psiZFilter = psiPost * s_zFilterPre[i - idPreStart];
            const float psiBetaEpsilonA = psiPost * beta * epsilonA;

            // Calculate e and episilonA
            const float e = psiZFilter - psiBetaEpsilonA;
            d_epsilonA[idSyn] = psiZFilter + ((rho * epsilonA) - psiBetaEpsilonA);

            // Calculate filtered version of eligibility trace
            float eFiltered = d_eFiltered[idSyn];
            eFiltered = (eFiltered * alpha) + e;

            // Apply weight update
            d_deltaG[idSyn] += (eFiltered * ePost) + ((fAvgPost - fTargetTimestep) * cReg * e);
            d_eFiltered[idSyn] = eFiltered;


            idSyn += numPost;
        }
    }
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
int main(int argc, char *argv[])
{
    unsigned int blockSize = 32;
    unsigned int numNeurons = 10000;
    unsigned int numTimesteps = 5000;

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
    if(argc > 4) {
        numTimesteps = std::stoul(argv[4]);
    }

    std::cout << "Model:" << static_cast<int>(model) << ", num neurons:" << numNeurons << ", block size:" << blockSize << std::endl;

    const unsigned int numSynapses = numNeurons * numNeurons;

    const unsigned int numNeuronBlocks = (numNeurons + blockSize - 1) / blockSize;

    CHECK_CUDA_ERRORS(cudaSetDevice(0));

    std::mt19937 rng;
    std::normal_distribution<float> dis(0.0f, 1.0f);

    if(model == Model::Continuous) {
        // Create array to hold postsynaptic output
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
            {
                Timer<std::milli> t("Continuous Dense:");
                const unsigned int numBlocks = (numSynapses + blockSize - 1) / blockSize;
                dim3 threads(blockSize, 1);
                dim3 grid(numBlocks, 1);

                for(unsigned int i = 0; i < numTimesteps; i++) {
                    continuousDense<<<grid, threads>>>(numNeurons, numNeurons, output.second, ePre.second, g.second);
                }
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }

        // Zero output
        std::fill_n(&output.first[0], numNeurons, 0.0f);
        hostToDeviceCopy(output, numNeurons);

        {
            {
                Timer<std::milli> t("Continuous Dense Block:");
                const unsigned int numBlocks = numNeuronBlocks * numNeuronBlocks;
                const unsigned int sharedBytes = blockSize * sizeof(unsigned int);
                dim3 threads(blockSize, 1);
                dim3 grid(numBlocks, 1);

                for(unsigned int i = 0; i < numTimesteps; i++) {
                    continuousDenseBlock<<<grid, threads, sharedBytes>>>(numNeurons, numNeurons, numNeuronBlocks * blockSize,
                                                                         output.second, ePre.second, g.second);
                }
            }

            deviceToHostCopy(output, numNeurons);
            const float sum = std::accumulate(&output.first[0], &output.first[numNeurons], 0.0f);
            std::cout << "Sum:" << sum << std::endl;
        }
    }
    else {
        // Create array to hold presynaptic variables
        auto zFilterPre = allocateHostDevice<float>(numNeurons);

        // Create arrays to hold postsynaptic variables
        auto psiPost = allocateHostDevice<float>(numNeurons);
        auto fAvgPost = allocateHostDevice<float>(numNeurons);
        auto ePost = allocateHostDevice<float>(numNeurons);
        
        // Create arrays to hold synaptic variables
        auto eFiltered = allocateHostDevice<float>(numSynapses);
        auto epsilonA = allocateHostDevice<float>(numSynapses);
        auto deltaG = allocateHostDevice<float>(numSynapses);
        
        // Randomize pre and postsynaptic state
        std::generate_n(&zFilterPre.first[0], numNeurons, [&rng, &dis]() { return dis(rng); });
        std::generate_n(&psiPost.first[0], numNeurons, [&rng, &dis]() { return dis(rng); });
        std::generate_n(&fAvgPost.first[0], numNeurons, [&rng, &dis]() { return dis(rng); });
        std::generate_n(&ePost.first[0], numNeurons, [&rng, &dis]() { return dis(rng); });
        hostToDeviceCopy(zFilterPre, numNeurons);
        hostToDeviceCopy(psiPost, numNeurons);
        hostToDeviceCopy(fAvgPost, numNeurons);
        hostToDeviceCopy(ePost, numNeurons);

        // Zero synaptic state
        std::fill_n(&eFiltered.first[0], numSynapses, 0.0f);
        std::fill_n(&epsilonA.first[0], numSynapses, 0.0f);
        std::fill_n(&deltaG.first[0], numSynapses, 0.0f);
        hostToDeviceCopy(eFiltered, numSynapses);
        hostToDeviceCopy(epsilonA, numSynapses);
        hostToDeviceCopy(deltaG, numSynapses);

        {
            {
                Timer<std::milli> t("ePropALIF Dense:");
                const unsigned int numBlocks = (numSynapses + blockSize - 1) / blockSize;
                dim3 threads(blockSize, 1);
                dim3 grid(numBlocks, 1);

                for(unsigned int i = 0; i < numTimesteps; i++) {
                    epropALIFDense<<<grid, threads>>>(numNeurons, numNeurons,
                                                      zFilterPre.second,  psiPost.second,
                                                      fAvgPost.second, ePost.second,
                                                      eFiltered.second, epsilonA.second, deltaG.second);
                }
            }

            deviceToHostCopy(eFiltered, numSynapses);
            deviceToHostCopy(epsilonA, numSynapses);
            deviceToHostCopy(deltaG, numSynapses);
            const float eFilteredSum = std::accumulate(&eFiltered.first[0], &eFiltered.first[numSynapses], 0.0f);
            std::cout << "eFiltered sum:" << eFilteredSum << std::endl;
            const float epsilonASum = std::accumulate(&epsilonA.first[0], &epsilonA.first[numSynapses], 0.0f);
            std::cout << "epsilonA sum:" << epsilonASum << std::endl;
            const float deltaGSum = std::accumulate(&deltaG.first[0], &deltaG.first[numSynapses], 0.0f);
            std::cout << "deltaG sum:" << deltaGSum << std::endl;
        }

         // Zero synaptic state
        std::fill_n(&eFiltered.first[0], numSynapses, 0.0f);
        std::fill_n(&epsilonA.first[0], numSynapses, 0.0f);
        std::fill_n(&deltaG.first[0], numSynapses, 0.0f);
        hostToDeviceCopy(eFiltered, numSynapses);
        hostToDeviceCopy(epsilonA, numSynapses);
        hostToDeviceCopy(deltaG, numSynapses);

        {
            {
                Timer<std::milli> t("ePropALIF Dense Block:");
                const unsigned int numBlocks = numNeuronBlocks * numNeuronBlocks;
                const unsigned int sharedBytes = blockSize * sizeof(unsigned int);
                dim3 threads(blockSize, 1);
                dim3 grid(numBlocks, 1);

                for(unsigned int i = 0; i < numTimesteps; i++) {
                    epropALIFDenseBlock<<<grid, threads, sharedBytes>>>(numNeurons, numNeurons, numNeuronBlocks * blockSize,
                                                                        zFilterPre.second,  psiPost.second,
                                                                        fAvgPost.second, ePost.second,
                                                                        eFiltered.second, epsilonA.second, deltaG.second);
                }
            }

            deviceToHostCopy(eFiltered, numSynapses);
            deviceToHostCopy(epsilonA, numSynapses);
            deviceToHostCopy(deltaG, numSynapses);
            const float eFilteredSum = std::accumulate(&eFiltered.first[0], &eFiltered.first[numSynapses], 0.0f);
            std::cout << "eFiltered sum:" << eFilteredSum << std::endl;
            const float epsilonASum = std::accumulate(&epsilonA.first[0], &epsilonA.first[numSynapses], 0.0f);
            std::cout << "epsilonA sum:" << epsilonASum << std::endl;
            const float deltaGSum = std::accumulate(&deltaG.first[0], &deltaG.first[numSynapses], 0.0f);
            std::cout << "deltaG sum:" << deltaGSum << std::endl;
        }
    
    }
    return EXIT_SUCCESS;
}
