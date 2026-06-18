// Headline classification — transformer (DistilBERT) inference path.
// 1) WordPiece-tokenize the headline (C++ tokenizer matching distilbert-base-uncased)
// 2) Run the INT8 ONNX graph via ONNX Runtime
// 3) Return P(fake) and a thresholded label.
//
// Keeps the classifier.hpp signatures so main.cpp / api.py are unchanged.

#include "classifier.hpp"

#include "tokenizer.hpp"
#include "onnx_model.hpp"

#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace
{
// Label thresholds (UI derives its own verdict from the score; these only set the
// string label printed by main.cpp / returned in the API "label" field).
constexpr double FAKE_THRESHOLD{0.60};
constexpr double REAL_THRESHOLD{0.40};

struct Config
{
    std::string onnxModel{"headline.int8.onnx"};
    std::string vocabFile{"vocab.txt"};
    int maxLen{64};
    int fakeIndex{1};
    bool doLower{true};
    bool stripAccents{true};
    std::string unk{"[UNK]"}, cls{"[CLS]"}, sep{"[SEP]"}, prefix{"##"};
};

std::unique_ptr<WordPieceTokenizer> g_tok;
std::unique_ptr<OnnxModel> g_model;
std::once_flag g_init;
int g_maxLen{64}; // set in initialize()

// one-entry memoization so main's classify+confidence calls run inference once
std::string g_lastText;
double g_lastProb{0.0};
bool g_haveLast{false};

bool fileExists(const std::string& p)
{
    std::ifstream f(p);
    return static_cast<bool>(f);
}

std::string slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Minimal scalar extraction for our fixed-format config.json (avoids a JSON dependency).
bool jsonRaw(const std::string& js, const std::string& key, std::string& out)
{
    const std::string needle = "\"" + key + "\"";
    size_t k = js.find(needle);
    if (k == std::string::npos) return false;
    size_t c = js.find(':', k + needle.size());
    if (c == std::string::npos) return false;
    size_t i = c + 1;
    while (i < js.size() && std::isspace(static_cast<unsigned char>(js[i]))) ++i;
    if (i >= js.size()) return false;
    if (js[i] == '"')
    {
        size_t e = js.find('"', i + 1);
        if (e == std::string::npos) return false;
        out = js.substr(i + 1, e - i - 1);
    }
    else
    {
        size_t e = i;
        while (e < js.size() && js[e] != ',' && js[e] != '}' &&
               js[e] != '\n' && js[e] != '\r')
            ++e;
        out = js.substr(i, e - i);
        while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    }
    return true;
}

void applyConfig(const std::string& js, Config& cfg)
{
    std::string v;
    if (jsonRaw(js, "onnx_model", v)) cfg.onnxModel = v;
    if (jsonRaw(js, "vocab_file", v)) cfg.vocabFile = v;
    if (jsonRaw(js, "max_len", v)) cfg.maxLen = std::stoi(v);
    if (jsonRaw(js, "fake_label_index", v)) cfg.fakeIndex = std::stoi(v);
    if (jsonRaw(js, "do_lower_case", v)) cfg.doLower = (v == "true");
    if (jsonRaw(js, "strip_accents", v)) cfg.stripAccents = (v == "true");
    if (jsonRaw(js, "unk_token", v)) cfg.unk = v;
    if (jsonRaw(js, "cls_token", v)) cfg.cls = v;
    if (jsonRaw(js, "sep_token", v)) cfg.sep = v;
    if (jsonRaw(js, "wordpiece_prefix", v)) cfg.prefix = v;
}

void initialize()
{
    // Resolve model directory (repo root vs build/ subdir), like the original code.
    std::string dir = fileExists("model/vocab.txt") ? "model/" : "../model/";

    Config cfg;
    const std::string cfgPath = dir + "config.json";
    if (fileExists(cfgPath)) applyConfig(slurp(cfgPath), cfg);
    g_maxLen = cfg.maxLen;

    g_tok = std::make_unique<WordPieceTokenizer>(dir + cfg.vocabFile, cfg.doLower,
                                                 cfg.stripAccents, cfg.unk, cfg.cls,
                                                 cfg.sep, cfg.prefix);
    g_model = std::make_unique<OnnxModel>(dir + cfg.onnxModel, cfg.fakeIndex);
}

double score(const std::string& headline)
{
    std::call_once(g_init, initialize);
    if (g_haveLast && headline == g_lastText) return g_lastProb;

    auto ids = g_tok->encode(headline, g_maxLen);
    double p = g_model->fakeProbability(ids);

    g_lastText = headline;
    g_lastProb = p;
    g_haveLast = true;
    return p;
}
} // namespace

int classify_headline(const std::string& headline)
{
    const double prob = score(headline);
    if (prob >= FAKE_THRESHOLD) return 1;
    if (prob <= REAL_THRESHOLD) return 0;
    return -1;
}

double confidence_headline(const std::string& headline)
{
    return score(headline);
}

std::vector<int64_t> headline_token_ids(const std::string& headline)
{
    std::call_once(g_init, initialize);
    return g_tok->encode(headline, g_maxLen);
}
