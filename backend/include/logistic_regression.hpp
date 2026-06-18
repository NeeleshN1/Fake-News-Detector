#pragma once

#include <vector>
#include <string>

class LogisticRegression
{
public:
    explicit LogisticRegression(size_t numFeatures, double lr = 0.01);

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs);

    double predictProbability(const std::vector<double>& x) const;
    int predict(const std::vector<double>& x) const;
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<double> weights;
    double bias;
    double learningRate;

    static double sigmoid(double z);
};
