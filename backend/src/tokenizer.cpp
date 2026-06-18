// WordPiece tokenizer implementation — mirrors HuggingFace BertTokenizer
// (do_lower_case=True) used by distilbert-base-uncased.
//
// Pipeline:
//   normalize()      -> clean control chars, strip accents (NFD+remove-Mn equivalent for Latin-1),
//                       lowercase ASCII, add spaces around CJK.
//   basicTokenize()  -> whitespace split, then split off punctuation as separate tokens.
//   wordpiece()      -> greedy longest-match subwording with "##" continuations, [UNK] on failure.

#include "tokenizer.hpp"

#include <fstream>
#include <stdexcept>

// ---------------- UTF-8 helpers ----------------

// Decode UTF-8 string into Unicode codepoints.
static std::vector<uint32_t> toCodepoints(const std::string& s)
{
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n)
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        int extra = 0;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; extra = 1; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; extra = 2; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; extra = 3; }
        else { cp = c; extra = 0; } // invalid lead byte -> treat as latin1
        ++i;
        for (int k = 0; k < extra && i < n; ++k, ++i)
        {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
        }
        cps.push_back(cp);
    }
    return cps;
}

// Encode a single codepoint as UTF-8 and append to out.
static void appendUtf8(std::string& out, uint32_t cp)
{
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp < 0x10000)
    {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Split a UTF-8 string into its component characters (each a UTF-8 substring).
static std::vector<std::string> toChars(const std::string& s)
{
    std::vector<std::string> out;
    auto cps = toCodepoints(s);
    out.reserve(cps.size());
    for (uint32_t cp : cps)
    {
        std::string c;
        appendUtf8(c, cp);
        out.push_back(std::move(c));
    }
    return out;
}

// ---------------- classification helpers ----------------

// Map a precomposed Latin-1 accented codepoint to its base ASCII letter, matching
// BERT's NFD-then-strip-combining-marks behaviour. Returns 0 if not an accented letter
// that decomposes (e.g. Æ, Ð, Ø, Þ, ß are kept as-is by BERT, so not mapped here).
static char stripLatin1Accent(uint32_t cp)
{
    switch (cp)
    {
        case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC4: case 0xC5:
        case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: return 'a';
        case 0xC7: case 0xE7: return 'c';
        case 0xC8: case 0xC9: case 0xCA: case 0xCB:
        case 0xE8: case 0xE9: case 0xEA: case 0xEB: return 'e';
        case 0xCC: case 0xCD: case 0xCE: case 0xCF:
        case 0xEC: case 0xED: case 0xEE: case 0xEF: return 'i';
        case 0xD1: case 0xF1: return 'n';
        case 0xD2: case 0xD3: case 0xD4: case 0xD5: case 0xD6:
        case 0xF2: case 0xF3: case 0xF4: case 0xF5: case 0xF6: return 'o';
        case 0xD9: case 0xDA: case 0xDB: case 0xDC:
        case 0xF9: case 0xFA: case 0xFB: case 0xFC: return 'u';
        case 0xDD: case 0xFD: case 0xFF: return 'y';
        default: return 0;
    }
}

static bool isWhitespace(uint32_t cp)
{
    // BERT treats \t \n \r and space as whitespace (and a few unicode spaces).
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return true;
    if (cp == 0x00A0 || cp == 0x2028 || cp == 0x2029) return true;
    return false;
}

static bool isControl(uint32_t cp)
{
    if (cp == '\t' || cp == '\n' || cp == '\r') return false; // handled as whitespace
    if (cp == 0) return true;
    // C0/C1 control ranges
    if (cp < 0x20) return true;
    if (cp >= 0x7F && cp <= 0x9F) return true;
    return false;
}

static bool isAsciiPunct(uint32_t cp)
{
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
}

// BERT considers ASCII punctuation plus any Unicode punctuation; we approximate the
// non-ASCII case with common punctuation blocks (sufficient for news headlines).
static bool isPunctuation(uint32_t cp)
{
    if (isAsciiPunct(cp)) return true;
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D) return true; // curly quotes
    if (cp == 0x2013 || cp == 0x2014) return true;                                  // en/em dash
    if (cp == 0x2026) return true;                                                  // ellipsis
    if (cp == 0x00A1 || cp == 0x00BF) return true;                                  // ¡ ¿
    return false;
}

static bool isCJK(uint32_t cp)
{
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

// ---------------- tokenizer ----------------

WordPieceTokenizer::WordPieceTokenizer(const std::string& vocabPath, bool doLowerCase,
                                       bool stripAccents, const std::string& unkToken,
                                       const std::string& clsToken, const std::string& sepToken,
                                       const std::string& wordpiecePrefix)
    : doLowerCase_(doLowerCase), stripAccents_(stripAccents),
      unk_(unkToken), cls_(clsToken), sep_(sepToken), prefix_(wordpiecePrefix)
{
    std::ifstream in(vocabPath, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open vocab file: " + vocabPath);

    std::string line;
    int64_t idx = 0;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // tolerate CRLF
        vocab_.emplace(line, idx++);
    }

    auto lookup = [&](const std::string& t) -> int64_t {
        auto it = vocab_.find(t);
        if (it == vocab_.end()) throw std::runtime_error("Vocab missing required token: " + t);
        return it->second;
    };
    unkId_ = lookup(unk_);
    clsId_ = lookup(cls_);
    sepId_ = lookup(sep_);
}

std::string WordPieceTokenizer::normalize(const std::string& text) const
{
    auto cps = toCodepoints(text);
    std::string out;
    out.reserve(text.size());
    for (uint32_t cp : cps)
    {
        if (cp == 0xFFFD || isControl(cp)) continue;     // drop replacement/control chars
        if (isWhitespace(cp)) { out.push_back(' '); continue; }

        if (stripAccents_)
        {
            char base = stripLatin1Accent(cp);
            if (base) { out.push_back(base); continue; } // already lowercase ASCII
        }
        if (doLowerCase_ && cp >= 'A' && cp <= 'Z') { out.push_back(static_cast<char>(cp + 32)); continue; }

        if (isCJK(cp)) { out.push_back(' '); appendUtf8(out, cp); out.push_back(' '); continue; }

        appendUtf8(out, cp);
    }
    return out;
}

std::vector<std::string> WordPieceTokenizer::basicTokenize(const std::string& text) const
{
    const std::string norm = normalize(text);
    std::vector<std::string> tokens;

    // whitespace split
    std::vector<std::string> words;
    std::string cur;
    for (char ch : norm)
    {
        if (ch == ' ')
        {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        }
        else cur.push_back(ch);
    }
    if (!cur.empty()) words.push_back(cur);

    // split punctuation off each word
    for (const auto& w : words)
    {
        auto chars = toChars(w);
        std::string piece;
        for (const auto& ch : chars)
        {
            uint32_t cp = toCodepoints(ch)[0];
            if (isPunctuation(cp))
            {
                if (!piece.empty()) { tokens.push_back(piece); piece.clear(); }
                tokens.push_back(ch);
            }
            else piece += ch;
        }
        if (!piece.empty()) tokens.push_back(piece);
    }
    return tokens;
}

std::vector<std::string> WordPieceTokenizer::wordpiece(const std::string& token) const
{
    auto chars = toChars(token);
    if (static_cast<int>(chars.size()) > maxCharsPerWord_)
        return {unk_};

    std::vector<std::string> output;
    size_t start = 0;
    const size_t n = chars.size();
    while (start < n)
    {
        size_t end = n;
        std::string curSub;
        bool found = false;
        while (start < end)
        {
            std::string sub;
            for (size_t k = start; k < end; ++k) sub += chars[k];
            if (start > 0) sub = prefix_ + sub;
            if (vocab_.find(sub) != vocab_.end()) { curSub = sub; found = true; break; }
            --end;
        }
        if (!found) return {unk_}; // whole word is OOV
        output.push_back(curSub);
        start = end;
    }
    return output;
}

std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& text) const
{
    std::vector<std::string> out;
    for (const auto& tok : basicTokenize(text))
        for (auto& sub : wordpiece(tok))
            out.push_back(std::move(sub));
    return out;
}

int64_t WordPieceTokenizer::id(const std::string& token) const
{
    auto it = vocab_.find(token);
    return it == vocab_.end() ? unkId_ : it->second;
}

std::vector<int64_t> WordPieceTokenizer::encode(const std::string& text, int maxLen) const
{
    auto toks = tokenize(text);
    const size_t maxBody = (maxLen > 2) ? static_cast<size_t>(maxLen - 2) : 0;
    if (toks.size() > maxBody) toks.resize(maxBody); // truncate to fit [CLS]/[SEP]

    std::vector<int64_t> ids;
    ids.reserve(toks.size() + 2);
    ids.push_back(clsId_);
    for (const auto& t : toks) ids.push_back(id(t));
    ids.push_back(sepId_);
    return ids;
}
