#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define BLOCK_SIZE 512

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Upsweep Kernel
__global__ void
upsweep_kernel(int* data, int N, int twod) {
    int twod1 = twod * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx * twod1;
    if (i < N)
        data[i+twod1-1] += data[i+twod-1];
    // On the last iteration, we set the last bit to zero. 
    if (twod * 2 == N && i == 0)
        data[N-1] = 0;
}

// Downsweep Kernel
__global__ void
downsweep_kernel(int* data, int N, int twod) {
    int twod1 = twod * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx * twod1;
    if (i < N) {
        int t = data[i+twod-1];
        data[i+twod-1] = data[i+twod1-1];
        data[i+twod1-1] += t;
    }
}


void exclusive_scan(int* device_data, int length)
{
    int rounded_length = nextPow2(length);
    
    // Upsweep Phase
    for (int twod = 1; twod < rounded_length; twod *= 2) {
        int twod1 = twod * 2;
        int totalCount = rounded_length / twod1;
        int blockCount = (totalCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        upsweep_kernel<<<blockCount, BLOCK_SIZE>>>(device_data, rounded_length, twod);
    }
    // Downsweep Phase
    for (int twod = rounded_length/2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int totalCount = rounded_length / twod1;
        int blockCount = (totalCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        downsweep_kernel<<<blockCount, BLOCK_SIZE>>>(device_data, rounded_length, twod);
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
peak_find_kernel(int *inarray, int N, int *resultarray) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N-1) {
        bool isPeak = inarray[i-1] < inarray[i] && inarray[i] > inarray[i+1];
        resultarray[i] = (isPeak) ? 1 : 0;
    }
}

__global__ void
peak_reduce_kernel(int *inarray, int N, int *resultarray) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i < N-1) && (inarray[i] != inarray[i+1])) 
        resultarray[inarray[i]] = i;
}


int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    int *peaks_array;
    int rounded_length = nextPow2(length);
    int blockCount = (rounded_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Map peaks to 1 and non-peaks to 0.
    cudaMalloc((void **)&peaks_array, rounded_length * sizeof(int));
    peak_find_kernel<<<blockCount, BLOCK_SIZE>>>(device_input, length, peaks_array);

    // Perform exclusive scan on peaks_array to collect results.
    exclusive_scan(peaks_array, length);
    peak_reduce_kernel<<<blockCount, BLOCK_SIZE>>>(peaks_array, length, device_output);

    // Retrieve return size, which is the last element of peaks_array.
    int return_size;
    cudaMemcpy(&return_size, peaks_array + (length - 1), sizeof(int),
        cudaMemcpyDeviceToHost);

    return return_size;
}


/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
