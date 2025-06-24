# CUDA Word Search (UTF-8 Friendly)

This is a **CUDA-accelerated word search application** designed to search for **simple, unbroken words** in **massive wordlists** quickly. The program is **UTF-8 friendly** and outputs two files:

- One with matching words only
- One with matching words and their occurrence counts

---

## ✅ Requirements

### ▶ Windows Users

**Optimal**:  
Use the **precompiled `.exe`** located in the `Precompiled/` folder.  
Requires **NVIDIA GPU** with CUDA support.

**Suboptimal**:  
Compile manually with a C++ compiler.  
The project uses **Premake** configured for:
- **Visual Studio Community 2022**
- **CUDA Toolkit 12.6**

#### 🔧 Acceptable Version Ranges

| Tool              | Acceptable Range            |
|-------------------|-----------------------------|
| Visual Studio     | 2019 (16.11) → 2022 (17.9)  |
| MSVC              | 14.29 → 14.39               |
| CUDA Toolkit      | 12.4, 12.5, 12.6            |
| Windows SDK       | 10.0.19041.0 or newer       |

---

## 🚀 Quick Start (Recommended)

### Using Precompiled Binary (Windows only)

1. Download `WordCounter.exe` from the `Precompiled/` folder.
2. Place your wordlists in the same folder.
3. Run via command line:

```bash
WordCounter.exe top10knames.txt rockyou.txt 500
```

- `top10knames.txt` → your small wordlist (e.g., found passwords or names)
- `rockyou.txt` → large dictionary to search through
- `500` → number of top matches to save (ranked by count)

**Output**:  
- `results.txt` → top 500 matching words  
- `results_with_counts.txt` → same list with occurrence counts

---

## ⚙️ Alternative: Compile with NVCC (Cross-Platform)

1. Go to the `src/nvcc/` directory.
2. Run:

```bash
nvcc -allow-unsupported-compiler -O2 -std=c++17 -o WordCounter.exe trie_counter_final_fixed.cu
```

- Works on both **Windows** and **Linux**.
- `trie_counter_final_fixed.cu` is provided as a fallback; it may not be as optimized but performs decently.

> ⚠️ This file may not be maintained as the project transitions to CMake and GUI support.

---

## 🛠 Advanced: Compile from Source

- Source files are located in the `src/` folder:
  - `wordcounter.cpp`
  - `kernel.cu` (or `trie_counter_final_fixed.cu` for NVCC route)

### ▶ On Windows

1. Adjust `premake5.lua` paths if needed.
2. Run Premake to generate the `.sln` file.
3. Open the solution in Visual Studio and build.

### ▶ On Linux

1. Modify `premake5.lua` for Linux paths.
2. Use a script or manually compile with `g++` and `nvcc`.
3. Run the compiled binary:

```bash
./WordCounter
```

---

## 📌 Notes

- GUI and cross-platform CMake-based builds are planned.
- Expect updates and improved performance in future versions.
- If you're new to Premake, using NVCC might be simpler for now.
