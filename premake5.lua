workspace "WordCounterCUDA"
    configurations { "Release", "Debug" }
    architecture "x64"
    startproject "WordCounter"

project "WordCounter"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "on" -- ✅ Use /MT or /MTd (static CRT)

    targetdir "bin/%{cfg.buildcfg}"
    objdir "build/obj/%{cfg.buildcfg}"

    files { "src/**.cpp", "src/**.cu", "src/**.h" }

    -- CUDA paths
    includedirs {
        "src",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
    }

    libdirs {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"
    }

    links { "cudart_static" }
    defines { "CUDA_STATIC" }

    -- 🔧 CUDA compile for Debug
    filter { "files:**.cu", "configurations:Debug" }
        buildmessage "Compiling CUDA kernel (Debug): %{file.name}"
        buildcommands {
            '"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -c -O0 -g -G -std=c++17 -Xcompiler "/EHsc /MTd /Zi" -D _DEBUG -D CUDA_STATIC -I src -o "%{cfg.objdir}/%{file.basename}.obj" "%{file.relpath}" -allow-unsupported-compiler'
        }
        buildoutputs {
            "%{cfg.objdir}/%{file.basename}.obj"
        }

    -- 🔧 CUDA compile for Release
    filter { "files:**.cu", "configurations:Release" }
        buildmessage "Compiling CUDA kernel (Release): %{file.name}"
        buildcommands {
            '"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -c -O2 -std=c++17 -Xcompiler "/EHsc /MT" -D NDEBUG -D CUDA_STATIC -I src -o "%{cfg.objdir}/%{file.basename}.obj" "%{file.relpath}" -allow-unsupported-compiler'
        }
        buildoutputs {
            "%{cfg.objdir}/%{file.basename}.obj"
        }

    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
