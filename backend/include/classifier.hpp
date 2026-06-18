#pragma once
#include <cstdint>
#include <string>
#include <vector>

int classify_headline(const std::string& headline);

double confidence_headline(const std::string& headline);

// Debug helper: returns the input_ids (incl. [CLS]/[SEP]) for parity testing
// against the Python tokenizer.
std::vector<int64_t> headline_token_ids(const std::string& headline);
