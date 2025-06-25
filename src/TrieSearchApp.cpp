#include "TrieSearchApp.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <cstdio>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#include <stdlib.h>
#endif

extern void launch_count_kernel(const std::string& wordlist_path,
                                const std::vector<std::string>& queries,
                                std::vector<std::pair<std::string, int>>& results);

std::string temp_path; // Track temp file for cleanup

static std::string save_stdin_to_temp_file() {
    std::string path;
#ifdef _WIN32
    char temp_file[MAX_PATH];
    if (GetTempFileNameA(".", "stdin", 0, temp_file) == 0) {
        std::cerr << "[ERROR] Failed to create temp file for stdin.\n";
        exit(1);
    }
    path = temp_file;
#else
    char tmp[] = "/tmp/stdinXXXXXX";
    int fd = mkstemp(tmp);
    if (fd == -1) {
        perror("mkstemp");
        exit(1);
    }
    path = tmp;
#endif

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] Failed to open temp file for writing.\n";
        exit(1);
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        out << line << '\n';
    }
    out.close();
    return path;
}

int TrieSearchApp::run(int argc, char** argv) {
    if (!parse_args(argc, argv))
        return 1;

    load_dictionary();
    search_and_print();
    return 0;
}

bool TrieSearchApp::parse_args(int argc, char** argv) {
    std::map<std::string, SearchType> modeMap = {
        {"Lookup", SearchType::Lookup},
        {"lenPrefix", SearchType::CommonPrefix},
        {"Prefix", SearchType::Predictive},
        {"Count", SearchType::Count}
    };

    cxxopts::Options options("WordCountCUDA", "usage WordCountCUDA.exe [options]");
    options.add_options()
        ("d,dict", "Dictionary file - what you want to search through", cxxopts::value<std::string>())
        ("m,mode", "Mode: Lookup | lenPrefix | Prefix | Count", cxxopts::value<std::string>()->default_value("Lookup"))
        ("q,query", "Query string or file (can do -q multiple times for small amount of single words)", cxxopts::value<std::vector<std::string>>())
        ("h,help", "Show this help message");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help") || argc == 1) {
            std::cout << "WordCountCUDA: usage WordCountCUDA.exe [options]\n";
            std::cout << "Options:\n";
            std::cout << "-d Dictionary file - what you want to search through\n";
            std::cout << "-q Query string or file (can do -q multiple times for small amount of single words)\n";
            std::cout << "-m Mode\n";
            std::cout << "  - `Lookup`: exact match\n";
            std::cout << "  - `lenPrefix`: common prefix search - prints exact length matches that appear. -q 123 will only output 123 if its found.\n";
            std::cout << "  - `Prefix`: predictive (autocomplete-style) - prints all that have this as a prefix -q 123 will output 123 1234 12345 123dog ect if found\n";
            std::cout << "  - `Count`: fast substring counter using GPU - outputs exact unbroken words in entire word. -q 123 will add to count for dog123 but not dog1@3\n\n";
            std::cout << "  application can accept stdin for dictionary -d option by using -d -\n";
            return false;
        }

        if ((!result.count("dict") || result["dict"].as<std::string>().empty()) ||
            (!result.count("query") || result["query"].as<std::vector<std::string>>().empty())) {
            std::cerr << "[WARNING] No lines or queries to process.\n";
            return false;
        }

        cfg.dictFile = result["dict"].as<std::string>();
        std::string modeStr = result["mode"].as<std::string>();
        if (!modeMap.count(modeStr)) {
            std::cerr << "[ERROR] Invalid mode: " << modeStr << "\n";
            return false;
        }
        cfg.mode = modeMap[modeStr];

        cfg.queries.clear();
        auto inputs = result["query"].as<std::vector<std::string>>();
        for (const auto& q : inputs) {
            std::ifstream test(q);
            if (test.good()) {
                std::string line;
                while (std::getline(test, line)) {
                    if (!line.empty())
                        cfg.queries.push_back(line);
                }
            } else {
                cfg.queries.push_back(q);
            }
        }

        if (cfg.queries.empty()) {
            std::cerr << "[WARNING] No valid query entries were loaded.\n";
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Argument parsing failed: " << e.what() << "\n";
        return false;
    }

    return true;
}

void TrieSearchApp::load_dictionary() {
    marisa::Keyset ks;
    std::ifstream file;
    std::string source_path;

    if (cfg.dictFile == "-") {
        temp_path = save_stdin_to_temp_file();
        cfg.dictFile = temp_path;
        source_path = temp_path;
    } else {
        source_path = cfg.dictFile;
    }

    file.open(source_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open dictionary file: " << source_path << "\n";
        exit(1);
    }

    std::string word;
    while (std::getline(file, word)) {
        if (!word.empty()) {
            word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());
            if (!word.empty())
                ks.push_back(word.c_str());
        }
    }

    if (ks.empty()) {
        std::cerr << "[WARNING] No lines were loaded from input.\n";
    }
    std::cerr << "[DEBUG] Loaded " << ks.size() << " words into trie.\n";
    trie.build(ks);
}

void TrieSearchApp::search_and_print() {
    for (const auto& query : cfg.queries) {
        std::vector<marisa::UInt32> ids;

        switch (cfg.mode) {
            case SearchType::Lookup: {
                marisa::Agent agent;
                agent.set_query(query);
                if (trie.lookup(agent)) {
                    ids.push_back(agent.key().id());
                }
                break;
            }
            case SearchType::CommonPrefix: {
                marisa::Agent agent;
                agent.set_query(query);
                while (trie.common_prefix_search(agent)) {
                    ids.push_back(agent.key().id());
                }
                break;
            }
            case SearchType::Predictive: {
                marisa::Agent agent;
                agent.set_query(query);
                while (trie.predictive_search(agent)) {
                    ids.push_back(agent.key().id());
                }
                break;
            }
            case SearchType::Count: {
                std::vector<std::pair<std::string, int>> results;
                launch_count_kernel(cfg.dictFile, cfg.queries, results);

                std::sort(results.begin(), results.end(), [](auto& a, auto& b) {
                    return b.second < a.second;
                });

                for (const auto& [word, count] : results) {
                    std::cout << word << " " << count << "\n";
                }

                if (!temp_path.empty()) {
                    std::remove(temp_path.c_str());
                    temp_path.clear();
                }
                return;
            }
        }

        marisa::Agent rev;
        for (auto id : ids) {
            rev.set_query(id);
            trie.reverse_lookup(rev);
            std::cout.write(rev.key().ptr(), rev.key().length());
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    if (!temp_path.empty()) {
        std::remove(temp_path.c_str());
        temp_path.clear();
    }
}
