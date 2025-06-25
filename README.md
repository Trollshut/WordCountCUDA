# WordCountCUDA

A high-performance, GPU-accelerated word search and counting tool using a MARISA trie and custom CUDA kernel.

This application is designed for lightning-fast scanning of massive wordlists (`-d`) against lists of search terms or queries (`-q`). It supports different trie-based search modes and a CUDA kernel mode for raw counting using literal substring matching.

---

## 🚀 Features

- **Trie-based** word lookup using MARISA
- **CUDA-accelerated** kernel for counting exact substring matches
- Supports multiple search modes:
  - `Lookup`: exact match
  - `lenPrefix`: common prefix search - prints exact length matches that appear. -q 123 will only output 123 if its found.
  - `Prefix`: predictive (autocomplete-style) - prints all that have this as a prefix -q 123 will output 123 1234 12345 123dog ect if found
  - `Count`: fast substring counter using GPU - outputs exact unbroken words in entire word. -q 123 will add to count for dog123 but not dog1@3
- Fully supports **UTF-8**
- Handles **very large input files**
- Outputs match results sorted by frequency (in `Count` mode)
- Really large sets of data will still be pretty slow as i havent optimized Count mode and the other modes are CPU driven as its a third party library.

---

## 📦 Requirements

### 🖥️ Windows (Precompiled EXE)
- Run directly using the `.exe` in `Precompiled/`

### 🔧 Building from Source
- Visual Studio 2022 (Community Edition)
- MSVC v14.29 – v14.39
- CUDA Toolkit 12.4 – 12.6
- CMake ≥ 3.18

> ❗️Note: Compiling with Visual Studio or `g++` is only for the host code. NVCC is **always used** to compile the `.cu` CUDA kernel regardless of how the host app is built.

---

## 🛠️ Build Instructions

```bash
build.bat
