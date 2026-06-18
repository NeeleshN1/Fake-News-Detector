// Model training pipeline
// This module is responsible for:
// 1) Loading data and building vocabulary
// 2) Training logistic regression model
// 3) Evaluate accuracy

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "dataset.hpp"
#include "text_preprocessing.hpp"
#include "vocabulary.hpp"
#include "vectorizer.hpp"
#include "logistic_regression.hpp"

// Trains and evaluates the fake news headline classifier
int main()
{
    try
    {
        // 1. Load dataset
        auto samples{loadDataset("../data/raw/headlines.csv")};

        std::cout << "Loaded " << samples.size() << " samples\n";

        // 2. Shuffle samples
        std::vector<size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<Sample> shuffledSamples{};
        shuffledSamples.reserve(samples.size());

        for (size_t idx : indices)
        {
            shuffledSamples.push_back(samples[idx]);
        }

        // 3. Train / validation split
        size_t splitIndex{static_cast<size_t>(0.8 * shuffledSamples.size())};

        std::vector<Sample> trainSamples(shuffledSamples.begin(), shuffledSamples.begin() + splitIndex);

        std::vector<Sample> valSamples(shuffledSamples.begin() + splitIndex, shuffledSamples.end());

        // 4. Build vocabulary (TRAIN ONLY)
        std::vector<std::vector<std::string>> trainTokens{};
        for (const auto& s : trainSamples)
        {
            trainTokens.push_back(preprocessHeadline(s.headline));
        }

        auto vocab{buildVocabulary(trainTokens, 3, 10000)};

        std::cout << "Vocab size: " << vocab.size() << "\n";

        // 5. Vectorize train / val
        std::vector<std::vector<double>> X_train{}, X_val{};
        std::vector<int> y_train{}, y_val{};

        for (const auto& s : trainSamples)
        {
            X_train.push_back(vectorizeHeadline(preprocessHeadline(s.headline), vocab, vocab.size()));
            y_train.push_back(s.label);
        }

        for (const auto& s : valSamples)
        {
            X_val.push_back(vectorizeHeadline(preprocessHeadline(s.headline), vocab, vocab.size()));
            y_val.push_back(s.label);
        }

        // 6. Train model
        LogisticRegression model(vocab.size(), 0.01);
        model.train(X_train, y_train, 5);

        // 7. Evaluate accuracy
        int correct{0};
        for (size_t i = 0; i < X_val.size(); ++i)
        {
            if (model.predict(X_val[i]) == y_val[i])
            {
                ++correct;
            }
        }

        double accuracy{static_cast<double>(correct) / X_val.size()};
        std::cout << "Validation accuracy: " << accuracy << "\n";

        // 8. SAVE artifacts
        model.save("model.bin");
        saveVocabulary(vocab, "vocab.txt");

        std::cout << "Saved model.bin and vocab.txt\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Training error: " << e.what() << "\n";
    }

    return 0;
}
