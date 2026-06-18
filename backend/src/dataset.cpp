// Dataset loading and CSV parsing
// This module is responsible for:
// 1) Reading labelled headline data into memory samples

#include "dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Trim leading and trailing whitespace (spaces, tabs, etc.)
static inline void trim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));

    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// Loads headline pairs from a csv dataset file
std::vector<Sample> loadDataset(const std::string& filepath)
{
    std::ifstream file(filepath);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open dataset: " + filepath);
    }

    std::vector<Sample> samples{};
    std::string line{};

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::string headline{};
        std::string labelStr{};

        // Case 1: headline is quoted (may contain commas)
        if (!line.empty() && line.front() == '"')
        {
            // Find the closing quote followed by a comma
            const std::string quoteComma{"\","};

            size_t endQuote = line.find(quoteComma);

            if (endQuote == std::string::npos)
            {
                continue; // malformed row
            }

            headline = line.substr(1, endQuote - 1);
            labelStr = line.substr(endQuote + quoteComma.size());
        }
        // Case 2: simple CSV row (no quotes)
        else
        {
            std::stringstream ss(line);
            std::getline(ss, headline, ',');
            std::getline(ss, labelStr);
        }

        trim(headline);
        trim(labelStr);

        if (headline.empty() || labelStr.empty())
        {
            continue;
        }

        int label{};

        try
        {
            label = std::stoi(labelStr);
        }
        catch (...)
        {
            continue; // skip bad rows
        }

        Sample sample{};
        sample.headline = headline;
        sample.label = label;

        samples.push_back(sample);
    }

    return samples;
}
