This is a CUDA accelerated Word search application. 
The intent is to search simple whole unbroken words in massive wordlists relativly quickly. (UTF-8 "Friendly") 
this program creates two output files, one just words and the other the same words with occourance count next to it.

Requirements:
Windows: Optimal -> use .exe in Precompiled folder.
		subOptimal->Compile using C++ compiler(premake is set up for Visual Studio community 2022 c++ desktop toolkit, CUDA 12.6 (defined in premake)
												
						 Acceptable Version Ranges for Windows Visual Studio Community
						 
						Tool	Acceptable Range
						
						Visual Studio	2019 (16.11) → 2022 (17.9)
						MSVC			14.29 → 14.39
						CUDA Toolkit	12.4, 12.5, 12.6
						Windows SDK		10.0.19041.0+





======================EASIEST WAY TO USE PROGRAM===================================
Precompiled Windows .exe available. You only require a NVIDIA GPU to use (Hopefully first kernel code with ai help 
so fingers crossed). 
===================================================================================






example usage:
dict - is your large dictionary (rockyou, weakpass, hashmobfounds,ect) 
wordlist - is your potfile founds or wordlist to count occourance of. 
int - is number of words and their counts to save. 500 = save 500 words and their counts of occourance.

 WordCount.exe wordlist dict int

WordCount.exe top10knames.txt rockyou.txt 500  <--- output files will have 500 words. 


Second Easiest Way, Compile with NVCC

use the CUDA toolkits nvcc tool to compile the .cu in src\nvcc using 
" nvcc -allow-unsupported-compiler -O2 -std=c++17 -o WordCounter.exe trie_counter_final_fixed.cu" 
while in the nvcc directory for both linux and windows. 

ill leave the trie_counter_final_fixed.cu file in the nvcc folder in the src folder but i probably wont maintain it as ill use cmake in the end. 
its not as optimized i dont think but it is decently fast and will for for a few days until i make the gui and cross platform support.
ok maybe a week or two. but still. might be easier than messing with premake if you havent messed with it before. otherwise...

Hardest wayish

for Linux the source code is here in the src folder. its just those two files the .cu and the .cpp.
only thing you should have to adjust is the premake for the directories for your files. 
currently premake is configured to build for windows. Just change premake settings and make a script for linux
for your c++ compiler (linux is typically g++ i think?)

after script runs for windows just open the .sln file with visual studio and build the application. 
for linux, itll just compile with g++ or whatever you use and just navigate to the build path and run it with ./whatyounamedit

	