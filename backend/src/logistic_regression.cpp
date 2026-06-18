// Logistic regression model implementation
// This module is responsible for:
// 1) Training
// 2) Inference
// 3) Model Serialization

#include "logistic_regression.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>

LogisticRegression::LogisticRegression(size_t numFeatures, double lr)
    : weights(numFeatures, 0.0), bias(0.0), learningRate(lr) {}

double LogisticRegression::sigmoid(double z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

double LogisticRegression::predictProbability(const std::vector<double>& x) const
{
    double z{bias};

    for (size_t i = 0; i < x.size(); ++i)
    {
        z += weights[i] * x[i];
    }
    return sigmoid(z);
}

int LogisticRegression::predict(const std::vector<double>& x) const
{
    if (predictProbability(x) >= 0.5)
    {
        return 1;
    }

    return 0;
}

void LogisticRegression::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < X.size(); ++i)
        {
            double y_hat{predictProbability(X[i])};
            double error{y_hat - y[i]};

            // Gradient descent update
            for (size_t j = 0; j < weights.size(); ++j)
            {
                weights[j] -= learningRate * error * X[i][j];
            }

            bias -= learningRate * error;
        }
    }
}

void LogisticRegression::save(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);

    if (!out) 
    {
        throw std::runtime_error("Failed to open model file for saving");
    }

    size_t n{weights.size()};

    out.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(weights.data()), n * sizeof(double));
    out.write(reinterpret_cast<const char*>(&bias), sizeof(double));
}

void LogisticRegression::load(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) 
    {
        throw std::runtime_error("Failed to open model file for loading");
    }

    size_t n{};

    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));

    weights.resize(n);

    in.read(reinterpret_cast<char*>(weights.data()), n * sizeof(double));

    in.read(reinterpret_cast<char*>(&bias), sizeof(double));
}
