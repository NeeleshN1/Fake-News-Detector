// ONNX Runtime inference wrapper for the fine-tuned DistilBERT headline classifier.
// Uses pImpl so onnxruntime headers stay out of the rest of the codebase.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class OnnxModel
{
public:
    // fakeIndex = logit index corresponding to the FAKE class (1 for our model).
    explicit OnnxModel(const std::string& modelPath, int fakeIndex = 1);
    ~OnnxModel();

    // Forward pass over a single sequence (attention_mask is all ones). Returns P(fake).
    double fakeProbability(const std::vector<int64_t>& inputIds) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int fakeIndex_;
};
