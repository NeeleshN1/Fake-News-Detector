#pragma once

#include <vector>
#include <string>
#include <cstddef>

#include "vocabulary.hpp"

std::vector<double>
vectorizeHeadline(const std::vector<std::string>& tokens, const Vocabulary& vocab, size_t vocabSize);

