#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#endif

// C-callable kernel launcher
extern "C" void launch_count_matches_kernel(
    const char* d_lines, const int* d_line_offsets, int num_lines,
    const char* d_query_data, const int* d_query_offsets, const int* d_query_lengths,
    int num_queries, int* d_counts
);

// Host-side wrapper
void launch_count_kernel(const std::string& wordlist_path,
                         const std::vector<std::string>& queries,
                         std::vector<std::pair<std::string, int>>& results) {
    std::vector<std::string> lines;

    // Windows-specific fix for binary mode
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif

    // Input from stdin or file
    if (wordlist_path == "-") {
        std::string line;
        while (std::getline(std::cin, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            if (!line.empty())
                lines.push_back(line);
        }
    } else {
        std::ifstream file(wordlist_path);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Could not open file: " << wordlist_path << "\n";
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            if (!line.empty())
                lines.push_back(line);
        }
    }

    if (lines.empty() || queries.empty()) {
        std::cerr << "[WARNING] No lines or queries to process.\n";
        results.clear();
        for (const auto& q : queries) {
            results.emplace_back(q, 0);
        }
        return;
    }

    // Flatten lines
    std::vector<char> flat_lines;
    std::vector<int> line_offsets = {0};
    for (const auto& l : lines) {
        flat_lines.insert(flat_lines.end(), l.begin(), l.end());
        flat_lines.push_back('\n');
        line_offsets.push_back(static_cast<int>(flat_lines.size()));
    }

    // Flatten queries
    std::vector<char> query_data;
    std::vector<int> query_offsets = {0}, query_lengths;
    for (const auto& q : queries) {
        query_data.insert(query_data.end(), q.begin(), q.end());
        query_lengths.push_back(static_cast<int>(q.size()));
        query_offsets.push_back(static_cast<int>(query_data.size()));
    }

    const int num_lines = static_cast<int>(lines.size());
    const int num_queries = static_cast<int>(queries.size());

    // Allocate and copy device buffers
    char* d_lines = nullptr;
    int* d_line_offsets = nullptr;
    char* d_query_data = nullptr;
    int* d_query_offsets = nullptr;
    int* d_query_lengths = nullptr;
    int* d_counts = nullptr;

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

    // Launch kernel
    launch_count_matches_kernel(
        d_lines,
        d_line_offsets,
        num_lines,
        d_query_data,
        d_query_offsets,
        d_query_lengths,
        num_queries,
        d_counts
    );

    std::vector<int> h_counts(num_queries);
    cudaMemcpy(h_counts.data(), d_counts, num_queries * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_lines);
    cudaFree(d_line_offsets);
    cudaFree(d_query_data);
    cudaFree(d_query_offsets);
    cudaFree(d_query_lengths);
    cudaFree(d_counts);

    // Build output
    results.clear();
    for (int i = 0; i < num_queries; ++i) {
        results.emplace_back(queries[i], h_counts[i]);
    }
}
