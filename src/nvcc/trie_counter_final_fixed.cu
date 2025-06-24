#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>

#define MAX_LEN 31
#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 250000

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
    if (global_idx >= num_lines + start_line) return;

    int offset = dict_offsets[global_idx];
    if (offset < 0) return;
    const char* line = dict_data + offset;

    int line_len = 0;
    while (line[line_len] != '\0') ++line_len;

    for (int i = 0; i < num_words; ++i) {
        const char* word = words + word_offsets[i];
        int wlen = word_lengths[i];

        if (line_len >= wlen) {
            for (int k = 0; k <= line_len - wlen; ++k) {
                int j = 0;
                while (j < wlen && line[k + j] == word[j]) ++j;
                if (j == wlen) {
                    atomicAdd(&word_counts[i], 1ULL);
                    break;
                }
            }
        }
    }
}

bool is_valid_utf8(const std::string& s) {
    int c, i, ix, n;
    for (i = 0, ix = s.length(); i < ix; i++) {
        c = (unsigned char)s[i];
        if (c <= 0x7f) n = 0;
        else if ((c & 0xE0) == 0xC0) n = 1;
        else if (c >= 0xF0) return false;
        else if ((c & 0xF0) == 0xE0) n = 2;
        else return false;
        for (int j = 0; j < n && i < ix; j++) {
            if ((++i == ix) || ((s[i] & 0xC0) != 0x80))
                return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " names.txt target.txt 1000\n";
        return 1;
    }

    std::ifstream namesFile(argv[1]), dictFile(argv[2], std::ios::binary);
    int result_limit = std::stoi(argv[3]);

    if (!namesFile || !dictFile) {
        std::cerr << "Failed to open input files.\n";
        return 1;
    }

    std::vector<std::string> words;
    std::string line;
    while (std::getline(namesFile, line)) {
        if (line.size() <= MAX_LEN && is_valid_utf8(line))
            words.push_back(line);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    dictFile.seekg(0, std::ios::end);
    size_t dict_size = dictFile.tellg();
    dictFile.seekg(0);

    std::string dict_blob(dict_size + 1, '\0');
    dictFile.read(&dict_blob[0], dict_size);

    std::vector<int> dict_offsets;
    dict_offsets.reserve(dict_size / 8);
    int offset = 0;
    for (size_t i = 0; i < dict_size; ++i) {
        if (dict_blob[i] == '\r') {
            dict_blob[i] = '\0';
            dict_offsets.push_back(offset);
            if (i + 1 < dict_size && dict_blob[i + 1] == '\n') ++i;
            offset = i + 1;
        } else if (dict_blob[i] == '\n') {
            dict_blob[i] = '\0';
            dict_offsets.push_back(offset);
            offset = i + 1;
        }
    }
    if (offset < dict_size) {
        dict_offsets.push_back(offset);
    }

    std::cout << "Total lines: " << dict_offsets.size()
              << ", Final offset: " << dict_offsets.back() << "\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Loaded %zu entries from %s in %.2fs\n",
           dict_offsets.size(), argv[2],
           std::chrono::duration<double>(t1 - t0).count());

    std::vector<int> word_offsets, word_lengths;
    std::string word_blob;
    int woff = 0;
    for (const auto& w : words) {
        word_offsets.push_back(woff);
        word_lengths.push_back((int)w.size());
        word_blob += w;
        woff += w.size();
    }

    char* d_dict;
    int* d_dict_offsets;
    char* d_words;
    int* d_word_offsets;
    int* d_word_lengths;
    unsigned long long* d_counts;

    cudaMalloc(&d_dict, dict_blob.size());
    cudaMalloc(&d_dict_offsets, dict_offsets.size() * sizeof(int));
    cudaMalloc(&d_words, word_blob.size());
    cudaMalloc(&d_word_offsets, word_offsets.size() * sizeof(int));
    cudaMalloc(&d_word_lengths, word_lengths.size() * sizeof(int));
    cudaMalloc(&d_counts, words.size() * sizeof(unsigned long long));
    cudaMemset(d_counts, 0, words.size() * sizeof(unsigned long long));

    cudaMemcpy(d_dict, dict_blob.data(), dict_blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dict_offsets, dict_offsets.data(), dict_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_words, word_blob.data(), word_blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_offsets, word_offsets.data(), word_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_lengths, word_lengths.data(), word_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    int total_lines = dict_offsets.size();
    int processed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (processed < total_lines) {
        int chunk = std::min(CHUNK_SIZE, total_lines - processed);
        int blocks = (chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        if (processed + chunk > total_lines) {
            std::cerr << "[FATAL] chunk boundary exceeds total_lines: "
                      << (processed + chunk) << " > " << total_lines << "\n";
            return 1;
        }

        count_matches_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_dict, d_dict_offsets, processed, chunk,
            d_words, d_word_offsets, d_word_lengths, d_counts, (int)words.size()
        );

        cudaDeviceSynchronize();
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernel_err) << "\n";
            return 1;
        }

        processed += chunk;

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        double pct = (double)processed / total_lines * 100.0;

        double remaining_sec = 0.0;
        if (pct > 0.0)
            remaining_sec = elapsed_sec * (100.0 / pct - 1.0);

        int hrs = static_cast<int>(remaining_sec / 3600);
        int mins = static_cast<int>((remaining_sec - hrs * 3600) / 60);
        int secs = static_cast<int>(remaining_sec) % 60;

        printf("[%.1f%%] Processed %d/%d | ETA: %02d:%02d:%02d\r",
               pct, processed, total_lines, hrs, mins, secs);
        std::cout << std::flush;
    }

    std::vector<unsigned long long> counts(words.size());
    cudaError_t copy_err = cudaMemcpy(counts.data(), d_counts, counts.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (copy_err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copy_err) << "\n";
        return 1;
    }

    std::vector<std::pair<std::string, unsigned long long>> sorted;
    for (size_t i = 0; i < words.size(); ++i) {
        if (counts[i] > 0)
            sorted.emplace_back(words[i], counts[i]);
    }

    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return b.second < a.second;
    });

    std::ofstream out1("matches_sorted.txt"), out2("matches_sorted_counts.txt");
    if (!out1 || !out2) {
        std::cerr << "Error: Failed to open output files for writing.\n";
        return 1;
    }

    int printed = 0;
    for (auto& [w, c] : sorted) {
        if (printed++ >= result_limit) break;
        out1 << w << "\n";
        out2 << w << " " << c << "\n";
    }

    std::cout << "Done. Outputs: matches_sorted.txt and matches_sorted_counts.txt\n";

    cudaFree(d_dict);
    cudaFree(d_dict_offsets);
    cudaFree(d_words);
    cudaFree(d_word_offsets);
    cudaFree(d_word_lengths);
    cudaFree(d_counts);

    return 0;
}