#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

using Vocabulary = std::unordered_map<std::string, size_t>;

// Build vocabulary from tokenized headlines
Vocabulary buildVocabulary(const std::vector<std::vector<std::string>>& tokenizedHeadlines,
    size_t minFrequency, size_t maxVocabSize);

// Save / load for training ↔ inference
void saveVocabulary(const Vocabulary& vocab, const std::string& path);
Vocabulary loadVocabulary(const std::string& path);


