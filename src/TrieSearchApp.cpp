#include "TrieSearchApp.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>

// Declare early to avoid "identifier not found" errors
extern void launch_count_kernel(const std::string& wordlist_path,
                                const std::vector<std::string>& queries,
                                std::vector<std::pair<std::string, int>>& results);

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

    cxxopts::Options options("TrieSearchApp", "Trie Search Tool");
    options.add_options()
        ("d,dict", "Dictionary file", cxxopts::value<std::string>())
        ("m,mode", "Search mode (Lookup, lenPrefix, Prefix, Count)", cxxopts::value<std::string>()->default_value("Lookup"))
        ("q,query", "Query string(s)", cxxopts::value<std::vector<std::string>>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("dict") || !result.count("query")) {
        std::cerr << options.help() << "\n";
        return false;
    }

    cfg.dictFile = result["dict"].as<std::string>();
    std::string modeStr = result["mode"].as<std::string>();
    if (!modeMap.count(modeStr)) {
        std::cerr << "Invalid mode: " << modeStr << "\n";
        return false;
    }
    cfg.mode = modeMap[modeStr];

    cfg.queries.clear();
    auto inputs = result["query"].as<std::vector<std::string>>();
    for (const auto& q : inputs) {
        std::ifstream fileCheck(q);
        if (fileCheck.good()) {
            std::string line;
            while (std::getline(fileCheck, line)) {
                if (!line.empty())
                    cfg.queries.push_back(line);
            }
        } else {
            cfg.queries.push_back(q);
        }
    }

    return true;
}

void TrieSearchApp::load_dictionary() {
    marisa::Keyset ks;
    std::ifstream dictIn(cfg.dictFile);
    std::string word;
    while (std::getline(dictIn, word)) {
        word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());
        if (!word.empty())
            ks.push_back(word.c_str());
    }

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
                return; // Skip MARISA output for Count mode
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
}