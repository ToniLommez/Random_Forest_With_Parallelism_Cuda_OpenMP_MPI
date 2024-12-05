#include "../include/tree.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <cuda/std/limits>
#include <iostream>

#ifdef CUDA

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void compute_gini_parallel(
    const float* d_X, const float* d_y,
    int num_samples, int num_features,
    float* d_best_impurity, int* d_best_feature, float* d_best_threshold) {

    // if(threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("device: num_samples: %d\ndevice: num_features: %d\n\n", num_samples, num_features);
    // }

    extern __shared__ float shared_memory[]; // For reduction
    float* shared_ginis = shared_memory;    // Gini values
    int* shared_features = (int*)&shared_ginis[blockDim.x]; // Features
    float* shared_thresholds = (float*)&shared_features[blockDim.x]; // Thresholds

    int feature = blockIdx.x; // Each block works on one feature
    int thread_idx = threadIdx.x;

    float best_local_impurity = cuda::std::numeric_limits<float>::max();
    float best_local_threshold = 0.0f;

    // Iterate over thresholds (samples) assigned to this thread
    for (int i = thread_idx; i < num_samples; i += blockDim.x) {
        float threshold = d_X[i * num_features + feature]; // Get threshold

        // Split data (device-side operations)
        int left_count = 0, right_count = 0;
        float left_gini = 0.0f, right_gini = 0.0f;

        for (int j = 0; j < num_samples; ++j) {
            float value = d_X[j * num_features + feature];
            if (value <= threshold) {
                left_count++;
                left_gini += d_y[j];
            } else {
                right_count++;
                right_gini += d_y[j];
            }
        }

        left_gini = (left_gini / left_count) * (1.0f - (left_gini / left_count));
        right_gini = (right_gini / right_count) * (1.0f - (right_gini / right_count));

        float weighted_impurity =
            (left_count * left_gini + right_count * right_gini) / num_samples;

        // Update local best impurity
        if (weighted_impurity < best_local_impurity) {
            best_local_impurity = weighted_impurity;
            best_local_threshold = threshold;
        }
    }

    // Store local results in shared memory
    shared_ginis[thread_idx] = best_local_impurity;
    shared_thresholds[thread_idx] = best_local_threshold;
    shared_features[thread_idx] = feature;

    __syncthreads();

    // Reduction: Find best impurity among threads in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            if (shared_ginis[thread_idx + stride] < shared_ginis[thread_idx]) {
                shared_ginis[thread_idx] = shared_ginis[thread_idx + stride];
                shared_thresholds[thread_idx] = shared_thresholds[thread_idx + stride];
            }
        }
        __syncthreads();
    }

    // Block-level result: store the best result of this block
    if (thread_idx == 0) {
        if (shared_ginis[0] < *d_best_impurity) {
            // atomicMinFloat(d_best_impurity, shared_ginis[0]);
            *d_best_impurity = shared_ginis[0];
            *d_best_feature = shared_features[0];
            *d_best_threshold = shared_thresholds[0];
        }
    }
}

void cuda_best_threshold_sender(
    const float_matrix& X, const float_vector& y,
    int num_samples, int num_features, int* best_feature, float* best_threshold) {

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    // Flatten X for device memory
    std::vector<float> X_flattened;
    for (const auto& row : X) {
        X_flattened.insert(X_flattened.end(), row.begin(), row.end());
    }

    // Allocate device memory
    float* d_X;
    float* d_y;
    float* d_best_impurity;
    int* d_best_feature;
    float* d_best_threshold;

    cudaMalloc(&d_X, num_samples * num_features * sizeof(float));
    cudaMalloc(&d_y, num_samples * sizeof(float));
    cudaMalloc(&d_best_impurity, sizeof(float));
    cudaMalloc(&d_best_feature, sizeof(int));
    cudaMalloc(&d_best_threshold, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, X_flattened.data(), num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), num_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize best impurity on the device
    float best_impurity = numeric_limits<float>::max();
    cudaMemcpy(d_best_impurity, &best_impurity, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 1024;
    int shared_memory_size = threads_per_block * (sizeof(float) + sizeof(int) + sizeof(float));
    // std::cout << "Shared memory calculated: " << shared_memory_size << " bytes" << std::endl;
    // std::cout << "host: num_samples: " << num_samples << std::endl;
    // std::cout << "host: num_features: " << num_features << std::endl;
    compute_gini_parallel<<<num_features, threads_per_block, shared_memory_size>>>(
        d_X, d_y, num_samples, num_features, d_best_impurity, d_best_feature, d_best_threshold);

    // Copy results back to host
    cudaMemcpy(best_feature, d_best_feature, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_threshold, d_best_threshold, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_best_impurity);
    cudaFree(d_best_feature);
    cudaFree(d_best_threshold);
}

#endif
