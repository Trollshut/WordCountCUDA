#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>

__device__ bool is_match(const char* haystack, int hay_len, const char* needle, int needle_len) {
    for (int i = 0; i <= hay_len - needle_len; ++i) {
        bool match = true;
        for (int j = 0; j < needle_len; ++j) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

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

void launch_count_kernel(const std::string& wordlist_path,
                         const std::vector<std::string>& queries,
                         std::vector<std::pair<std::string, int>>& results) {
    std::ifstream infile(wordlist_path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    std::vector<char> flat_lines;
    std::vector<int> line_offsets = {0};
    for (const auto& l : lines) {
        flat_lines.insert(flat_lines.end(), l.begin(), l.end());
        line_offsets.push_back(static_cast<int>(flat_lines.size()));
    }

    std::vector<char> query_data;
    std::vector<int> query_offsets = {0}, query_lengths;
    for (const auto& q : queries) {
        query_data.insert(query_data.end(), q.begin(), q.end());
        query_offsets.push_back(static_cast<int>(query_data.size()));
        query_lengths.push_back(static_cast<int>(q.size()));
    }

    char* d_lines = nullptr;
    int* d_line_offsets = nullptr;
    char* d_query_data = nullptr;
    int* d_query_offsets = nullptr;
    int* d_query_lengths = nullptr;
    int* d_counts = nullptr;

    int num_lines = static_cast<int>(lines.size());
    int num_queries = static_cast<int>(queries.size());
    std::vector<int> h_counts(num_queries, 0);

    cudaMalloc(&d_lines, flat_lines.size());
    cudaMemcpy(d_lines, flat_lines.data(), flat_lines.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_line_offsets, line_offsets.size() * sizeof(int));
    cudaMemcpy(d_line_offsets, line_offsets.data(), line_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_query_data, query_data.size());
    cudaMemcpy(d_query_data, query_data.data(), query_data.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_query_offsets, query_offsets.size() * sizeof(int));
    cudaMemcpy(d_query_offsets, query_offsets.data(), query_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_query_lengths, query_lengths.size() * sizeof(int));
    cudaMemcpy(d_query_lengths, query_lengths.data(), query_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_counts, num_queries * sizeof(int));
    cudaMemset(d_counts, 0, num_queries * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (num_lines + threadsPerBlock - 1) / threadsPerBlock;

    count_matches_kernel<<<blocks, threadsPerBlock>>>(
        d_lines, d_line_offsets, num_lines,
        d_query_data, d_query_offsets, d_query_lengths,
        num_queries, d_counts
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_counts.data(), d_counts, num_queries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_lines);
    cudaFree(d_line_offsets);
    cudaFree(d_query_data);
    cudaFree(d_query_offsets);
    cudaFree(d_query_lengths);
    cudaFree(d_counts);

    results.clear();
    for (size_t i = 0; i < queries.size(); ++i) {
        results.emplace_back(queries[i], h_counts[i]);
    }
}
