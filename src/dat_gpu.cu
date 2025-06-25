#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Matching substrings without overlaps
__device__ bool is_match(const char* haystack, int hay_len, const char* needle, int needle_len) {
    for (int i = 0; i <= hay_len - needle_len; ++i) {
        bool match = true;
        for (int j = 0; j < needle_len; ++j) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Main GPU kernel
__global__ void count_matches_kernel(const char* lines, const int* line_offsets, int num_lines,
                                     const char* query_data, const int* query_offsets, const int* query_lengths,
                                     int num_queries, int* counts) {
    int line_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (line_id >= num_lines) return;

    const char* line = lines + line_offsets[line_id];
    int line_len = line_offsets[line_id + 1] - line_offsets[line_id];

    for (int q = 0; q < num_queries; ++q) {
        const char* query = query_data + query_offsets[q];
        int query_len = query_lengths[q];

        if (line_len >= query_len && is_match(line, line_len, query, query_len)) {
            atomicAdd(&counts[q], 1);
        }
    }
}

// C++ callable kernel launcher
extern "C" void launch_count_matches_kernel(
    const char* d_lines, const int* d_line_offsets, int num_lines,
    const char* d_query_data, const int* d_query_offsets, const int* d_query_lengths,
    int num_queries, int* d_counts
) {
    int threadsPerBlock = 256;
    int blocks = (num_lines + threadsPerBlock - 1) / threadsPerBlock;

    count_matches_kernel<<<blocks, threadsPerBlock>>>(
        d_lines, d_line_offsets, num_lines,
        d_query_data, d_query_offsets, d_query_lengths,
        num_queries, d_counts
    );

    cudaDeviceSynchronize();
}
