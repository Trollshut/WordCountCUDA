// TrieSearchApp.hpp
#pragma once

#include "cxxopts.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <marisa.h>
#include <cuda_runtime.h>

enum class SearchType { Lookup, CommonPrefix, Predictive, Count};

class TrieSearchApp {
public:
    int run(int argc, char** argv);

private:
    struct Config {
        std::string dictFile;
        SearchType mode{ SearchType::Lookup };
        std::vector<std::string> queries;
    };

    Config cfg;
    marisa::Trie trie;

    bool parse_args(int argc, char** argv);
    void load_dictionary();
    void search_and_print();
};
