#pragma once

#include <string>
#include <vector>

struct Sample
{
    std::string headline;
    int label; // 0 = real, 1 = fake
};

std::vector<Sample> loadDataset(const std::string& filepath);
