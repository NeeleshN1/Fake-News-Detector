// ONNX Runtime inference implementation. Loads the (INT8) DistilBERT graph and runs a
// single-sequence forward pass, returning P(fake) via softmax over the 2 logits.

#include "onnx_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#ifdef _WIN32
#include <string>
static std::wstring toOrtPath(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#define ORT_PATH(s) toOrtPath(s).c_str()
#else
#define ORT_PATH(s) (s).c_str()
#endif

struct OnnxModel::Impl
{
    Ort::Env env;
    Ort::SessionOptions opts;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions alloc;
    std::string inIds, inMask, outName;

    explicit Impl(const std::string& path)
        : env(ORT_LOGGING_LEVEL_WARNING, "fake_news"),
          opts(),
          session(nullptr)
    {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, ORT_PATH(path), opts);

        // Resolve I/O names (export uses input_ids, attention_mask, logits).
        auto in0 = session.GetInputNameAllocated(0, alloc);
        auto in1 = session.GetInputNameAllocated(1, alloc);
        auto out0 = session.GetOutputNameAllocated(0, alloc);
        inIds = in0.get();
        inMask = in1.get();
        outName = out0.get();
    }
};

OnnxModel::OnnxModel(const std::string& modelPath, int fakeIndex)
    : impl_(std::make_unique<Impl>(modelPath)), fakeIndex_(fakeIndex) {}

OnnxModel::~OnnxModel() = default;

double OnnxModel::fakeProbability(const std::vector<int64_t>& inputIds) const
{
    const int64_t len = static_cast<int64_t>(inputIds.size());
    std::array<int64_t, 2> shape{1, len};

    std::vector<int64_t> ids(inputIds);
    std::vector<int64_t> mask(static_cast<size_t>(len), 1);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tIds = Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(),
                                                        shape.data(), shape.size());
    Ort::Value tMask = Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(),
                                                         shape.data(), shape.size());

    const char* inputNames[] = {impl_->inIds.c_str(), impl_->inMask.c_str()};
    const char* outputNames[] = {impl_->outName.c_str()};
    Ort::Value inputs[] = {std::move(tIds), std::move(tMask)};

    auto out = impl_->session.Run(Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);

    const float* logits = out[0].GetTensorData<float>();
    // softmax over 2 logits
    const float m = std::max(logits[0], logits[1]);
    const double e0 = std::exp(static_cast<double>(logits[0] - m));
    const double e1 = std::exp(static_cast<double>(logits[1] - m));
    const double pFake = (fakeIndex_ == 1) ? e1 / (e0 + e1) : e0 / (e0 + e1);
    return pFake;
}
