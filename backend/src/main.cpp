// This module is responsible for:
// 1) Providing command line inference
// 2) Interactive mode for local tests
//
// On Windows, argv arrives in the system ANSI codepage, which corrupts non-ASCII
// headlines before the (UTF-8) tokenizer sees them. We therefore read the wide
// command line and convert to UTF-8 so Unicode headlines tokenize correctly
// (Linux/Docker already delivers UTF-8 argv).

#include <iostream>
#include <string>
#include <vector>

#include "classifier.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "Shell32.lib")

static std::vector<std::string> get_args(int, char**)
{
    int n = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> args;
    for (int i = 0; i < n; ++i)
    {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(len > 0 ? len - 1 : 0, '\0');
        if (len > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], len, nullptr, nullptr);
        args.push_back(std::move(s));
    }
    if (wargv) LocalFree(wargv);
    return args;
}
#else
static std::vector<std::string> get_args(int argc, char** argv)
{
    return std::vector<std::string>(argv, argv + argc);
}
#endif

int main(int argc, char* argv[])
{
    const std::vector<std::string> args{get_args(argc, argv)};

    // DEBUG MODE: print token ids for parity testing  (./fake_news --ids "headline")
    if (args.size() > 2 && args[1] == "--ids")
    {
        for (int64_t id : headline_token_ids(args[2]))
        {
            std::cout << id << ' ';
        }
        std::cout << std::endl;
        return 0;
    }

    // BATCH MODE: read headlines from stdin (one per line), print "LABEL\tprob" per line.
    // The model loads once (call_once), so N headlines cost a single model load.
    // Reading from stdin (vs argv) also keeps UTF-8 intact on Windows.
    if (args.size() > 1 && args[1] == "--batch")
    {
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty())
            {
                std::cout << "UNDETERMINED\t0.5\n";
                continue;
            }
            int label{classify_headline(line)};
            double prob{confidence_headline(line)};
            const char* l{(label == 1) ? "FAKE" : (label == 0) ? "REAL" : "UNDETERMINED"};
            std::cout << l << '\t' << prob << '\n';
        }
        std::cout.flush();
        return 0;
    }

    // NON-INTERACTIVE MODE (API / Docker)
    if (args.size() > 1)
    {
        const std::string& headline{args[1]};

        int label{classify_headline(headline)};
        double prob{confidence_headline(headline)};

        if (label == 1)
        {
            std::cout << "FAKE " << prob << std::endl;
        }
        else if (label == 0)
        {
            std::cout << "REAL " << prob << std::endl;
        }
        else
        {
            std::cout << "UNDETERMINED " << prob << std::endl;
        }

        return 0;
    }

    // INTERACTIVE MODE (LOCAL TESTING)
    std::cout << "Fake News Headline Detector (Native)\n";
    std::cout << "----------------------------------\n";
    std::cout << "Enter a headline (empty line to quit):\n";

    std::string input{};
    std::getline(std::cin >> std::ws, input);

    while (!input.empty())
    {
        int label{classify_headline(input)};
        double prob{confidence_headline(input)};

        std::cout << "\nPrediction: ";
        if (label == 1)
        {
            std::cout << "FAKE";
        }
        else if (label == 0)
        {
            std::cout << "REAL";
        }
        else
        {
            std::cout << "UNDETERMINED";
        }

        std::cout << "\nConfidence: " << prob << "\n\n";
        std::cout << "Enter another headline:\n";

        std::getline(std::cin, input);
    }

    return 0;
}
