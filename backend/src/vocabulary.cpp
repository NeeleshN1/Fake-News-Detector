// Vocabulary construction
// This module is responsible for:
// 1) Building a word to index mapping
// 2) Saving/loading from disk

#include "vocabulary.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

// Builds a vocabulary from tokenized headlines with frequency filtering
Vocabulary buildVocabulary(const std::vector<std::vector<std::string>>& tokenizedHeadlines, size_t minFrequency,
    size_t maxVocabSize)
{
    // Count word frequencies
    std::unordered_map<std::string, size_t> wordCounts{};

    for (const auto& tokens : tokenizedHeadlines)
    {
        for (const auto& word : tokens)
        {
            ++wordCounts[word];
        }
    }

    std::vector<std::pair<std::string, size_t>> sortedWords{};

    for (const auto& [word, count] : wordCounts)
    {
        if (count >= minFrequency)
        {
            sortedWords.emplace_back(word, count);
        }
    }

    // Sort by descending frequency
    std::sort(sortedWords.begin(), sortedWords.end(), [](const auto& a, const auto& b){ return a.second > b.second;});

    Vocabulary vocab{};
    size_t index{0};

    for (const auto& [word, _] : sortedWords)
    {
        if (vocab.size() >= maxVocabSize)
        {
            break;
        }

        vocab[word] = index++;
    }

    return vocab;
}

// Saves the vocabulary mapping to disk
void saveVocabulary(const Vocabulary& vocab, const std::string& path)
{
    std::ofstream out(path);

    if (!out)
    {
        throw std::runtime_error("Failed to open vocab file for saving");
    }

    for (const auto& [token, index] : vocab)
    {
        out << token << " " << index << "\n";
    }
}

// Loads a vocabulary mapping from disk
Vocabulary loadVocabulary(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open vocab file for loading");
    }

    Vocabulary vocab{};
    std::string token{};
    size_t index{};

    while (in >> token >> index)
    {
        vocab[token] = index;
    }

    return vocab;
}
