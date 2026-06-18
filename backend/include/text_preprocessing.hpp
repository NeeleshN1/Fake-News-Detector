#pragma once

#include <string>
#include <vector>

std::string normalizeText(const std::string& text);
std::vector<std::string> tokenize(const std::string& text);
std::vector<std::string> preprocessHeadline(const std::string& headline);
