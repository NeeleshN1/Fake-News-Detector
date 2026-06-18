// WordPiece tokenizer (C++) — replicates HuggingFace `distilbert-base-uncased`
// (BERT BasicTokenizer + WordPiece). Produces input_ids with [CLS]/[SEP],
// truncated to max_len. Used by the ONNX inference path.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class WordPieceTokenizer
{
public:
    explicit WordPieceTokenizer(const std::string& vocabPath,
                                bool doLowerCase = true,
                                bool stripAccents = true,
                                const std::string& unkToken = "[UNK]",
                                const std::string& clsToken = "[CLS]",
                                const std::string& sepToken = "[SEP]",
                                const std::string& wordpiecePrefix = "##");

    // Full encode: [CLS] + wordpiece(text) + [SEP], truncated so total length <= maxLen.
    std::vector<int64_t> encode(const std::string& text, int maxLen) const;

    // Wordpiece tokens only (no special tokens) — exposed for parity/debugging.
    std::vector<std::string> tokenize(const std::string& text) const;

    int64_t id(const std::string& token) const;
    size_t vocabSize() const { return vocab_.size(); }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    bool doLowerCase_;
    bool stripAccents_;
    std::string unk_, cls_, sep_, prefix_;
    int64_t unkId_{}, clsId_{}, sepId_{};
    int maxCharsPerWord_{100};

    // Pipeline stages (mirror BERT).
    std::string normalize(const std::string& text) const;          // clean + lower + strip accents + CJK spacing
    std::vector<std::string> basicTokenize(const std::string& text) const;
    std::vector<std::string> wordpiece(const std::string& token) const;
};
