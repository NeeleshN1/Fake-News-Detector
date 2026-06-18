// Text preprocessing and utilities
// This module is responsible for:
// 1) Normalizes and tokenizes raw headline for model input

#include "text_preprocessing.hpp"

#include <cctype>
#include <sstream>

// Converts text to lowercase and replaces non-alphabetic characters with spaces
std::string normalizeText(const std::string& text)
{
    std::string cleaned{};
    cleaned.reserve(text.size());

    for (char c : text)
    {
        if (std::isalpha(static_cast<unsigned char>(c)))
        {
            cleaned.push_back(static_cast<char>(std::tolower(c)));
        }
        else
        {
            cleaned.push_back(' ');
        }
    }

    return cleaned;
}

// Splits normalized text into individual word tokens
std::vector<std::string> tokenize(const std::string& text)
{
    std::vector<std::string> tokens{};
    std::stringstream ss{text};
    std::string token{};

    while (ss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<std::string> preprocessHeadline(const std::string& headline)
{
    const std::string normalized{normalizeText(headline)};
    return tokenize(normalized);
}
