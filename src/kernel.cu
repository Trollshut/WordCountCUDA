#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_LEN 31
#define THREADS_PER_BLOCK 256

__device__ int device_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0' && len < MAX_LEN) ++len;
    return len;
}

__global__ void count_matches_kernel(
    const char* __restrict__ dict_data,
    const int* dict_offsets,
    int start_line,
    int num_lines,
    const char* __restrict__ words,
    const int* word_offsets,
    const int* word_lengths,
    unsigned long long* word_counts,
    int num_words
) {
    int global_idx = start_line + blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= start_line + num_lines) return;

    int offset = dict_offsets[global_idx];
    if (offset < 0) return;

    const char* line = dict_data + offset;

    int line_len = 0;
    while (line[line_len] != '\0' && line_len < MAX_LEN) ++line_len;
    if (line_len == 0) return;

    // For each word
    for (int i = 0; i < num_words; ++i) {
        const char* word = words + word_offsets[i];
        int wlen = word_lengths[i];
        if (wlen == 0 || wlen > MAX_LEN) continue;

        // Match within line
        if (line_len >= wlen) {
            for (int k = 0; k <= line_len - wlen; ++k) {
                int j = 0;
                while (j < wlen && line[k + j] == word[j]) ++j;

                if (j == wlen) {
                    atomicAdd(&word_counts[i], 1ULL);
                    // Debug output: one line per match
                    // printf("Matched word[%d] on line %d\n", i, global_idx);
                    break; // one match per line max
                }
            }
        }
    }
}

extern "C" void launch_count_matches_kernel(
    const char* d_dict,
    const int* d_dict_offsets,
    int start_line,
    int num_lines,
    const char* d_words,
    const int* d_word_offsets,
    const int* d_word_lengths,
    unsigned long long* d_counts,
    int num_words
) {
    int blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_matches_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_dict, d_dict_offsets, start_line, num_lines,
        d_words, d_word_offsets, d_word_lengths, d_counts, num_words
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
    }
}
