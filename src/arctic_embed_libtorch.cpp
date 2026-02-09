// Arctic Embed Tiny - LibTorch Implementation
// Uses PyTorch C++ API with MPS GPU acceleration
// Modes: --json (output embedding as JSON array), default (benchmark)
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iomanip>

// ============================================================================
// WordPiece Tokenizer
// ============================================================================

class WordPieceTokenizer {
private:
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t cls_id_ = 101;   // [CLS]
    int64_t sep_id_ = 102;   // [SEP]
    int64_t unk_id_ = 100;   // [UNK]
    int max_input_chars_ = 200;
    int max_seq_len_ = 512;

public:
    bool load(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) return false;

        std::string line;
        int64_t idx = 0;
        while (std::getline(file, line)) {
            // Strip trailing \r for Windows-style line endings
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            vocab_[line] = idx++;
        }
        return !vocab_.empty();
    }

    // Basic text normalization: lowercase + strip accents + split on whitespace/punct
    std::vector<std::string> basicTokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current;

        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = text[i];
            if (c <= 0x20 || c == 0x7F) {
                // Whitespace/control: flush current token
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
            } else if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
                       (c >= '[' && c <= '`') || (c >= '{' && c <= '~')) {
                // Punctuation: flush + add as separate token
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                current += static_cast<char>(c);
                tokens.push_back(current);
                current.clear();
            } else {
                // Lowercase ASCII
                if (c >= 'A' && c <= 'Z') {
                    current += static_cast<char>(c + 32);
                } else {
                    current += static_cast<char>(c);
                }
            }
        }
        if (!current.empty()) {
            tokens.push_back(current);
        }
        return tokens;
    }

    // WordPiece subword tokenization
    std::vector<int64_t> wordPieceTokenize(const std::string& word) {
        if (word.size() > static_cast<size_t>(max_input_chars_)) {
            return {unk_id_};
        }

        std::vector<int64_t> output_ids;
        size_t start = 0;

        while (start < word.size()) {
            size_t end = word.size();
            int64_t found_id = -1;

            while (start < end) {
                std::string substr;
                if (start > 0) {
                    substr = "##" + word.substr(start, end - start);
                } else {
                    substr = word.substr(start, end - start);
                }

                auto it = vocab_.find(substr);
                if (it != vocab_.end()) {
                    found_id = it->second;
                    break;
                }
                --end;
            }

            if (found_id == -1) {
                output_ids.push_back(unk_id_);
                break;
            }

            output_ids.push_back(found_id);
            start = end;
        }

        return output_ids;
    }

    std::pair<std::vector<int64_t>, std::vector<int64_t>> tokenize(const std::string& text) {
        auto words = basicTokenize(text);

        std::vector<int64_t> input_ids;
        input_ids.push_back(cls_id_);

        for (const auto& word : words) {
            auto subword_ids = wordPieceTokenize(word);
            for (auto id : subword_ids) {
                if (static_cast<int>(input_ids.size()) >= max_seq_len_ - 1) break;
                input_ids.push_back(id);
            }
            if (static_cast<int>(input_ids.size()) >= max_seq_len_ - 1) break;
        }

        input_ids.push_back(sep_id_);

        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        return {input_ids, attention_mask};
    }
};

// ============================================================================
// Arctic Embed Model
// ============================================================================

class ArcticEmbedLibTorch {
private:
    torch::jit::script::Module model_;
    torch::Device device_;

public:
    ArcticEmbedLibTorch(const std::string& model_path, bool quiet = false)
        : device_(torch::kMPS) {

        if (!quiet) {
            std::cerr << "Loading model on MPS..." << std::endl;
        }

        try {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<float> embed(const std::vector<int64_t>& input_ids,
                             const std::vector<int64_t>& attention_mask) {
        torch::NoGradGuard no_grad;

        auto ids_tensor = torch::from_blob(
            const_cast<int64_t*>(input_ids.data()),
            {1, static_cast<int64_t>(input_ids.size())},
            torch::kLong
        ).clone().to(device_);

        auto mask_tensor = torch::from_blob(
            const_cast<int64_t*>(attention_mask.data()),
            {1, static_cast<int64_t>(attention_mask.size())},
            torch::kLong
        ).clone().to(device_);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(ids_tensor);
        inputs.push_back(mask_tensor);

        auto output_dict = model_.forward(inputs).toGenericDict();
        auto last_hidden_state = output_dict.at("last_hidden_state").toTensor();

        // Mean pooling
        auto pooled = last_hidden_state.mean(1).squeeze(0);

        // L2 normalize
        auto norm = pooled.norm(2);
        auto normalized = pooled / norm;

        auto cpu_tensor = normalized.to(torch::kCPU);
        auto data_ptr = cpu_tensor.data_ptr<float>();

        return std::vector<float>(data_ptr, data_ptr + cpu_tensor.numel());
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_text> [--json] [--vocab <path>]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_text = argv[2];

    bool json_mode = false;
    std::string vocab_path;

    // Parse optional flags
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            json_mode = true;
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        }
    }

    // Auto-detect vocab path if not specified
    if (vocab_path.empty()) {
        // Try relative to binary location
        std::string binary_path = argv[0];
        auto last_slash = binary_path.rfind('/');
        if (last_slash != std::string::npos) {
            vocab_path = binary_path.substr(0, last_slash) + "/vocab.txt";
        } else {
            vocab_path = "bin/vocab.txt";
        }
    }

    try {
        // Load tokenizer
        WordPieceTokenizer tokenizer;
        if (!tokenizer.load(vocab_path)) {
            std::cerr << "Failed to load vocab from: " << vocab_path << std::endl;
            return 1;
        }

        auto [input_ids, attention_mask] = tokenizer.tokenize(input_text);

        if (json_mode) {
            // JSON mode: output embedding array and exit
            ArcticEmbedLibTorch embedder(model_path, true);

            // One warmup run
            embedder.embed(input_ids, attention_mask);

            auto embedding = embedder.embed(input_ids, attention_mask);

            // Output as JSON array
            std::cout << "[";
            for (size_t i = 0; i < embedding.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << std::setprecision(8) << embedding[i];
            }
            std::cout << "]" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            // Benchmark mode
            std::cout << "==================================================" << std::endl;
            std::cout << "Arctic Embed - LibTorch (PyTorch C++) Version" << std::endl;
            std::cout << "==================================================" << std::endl;
            std::cout << std::endl;

            ArcticEmbedLibTorch embedder(model_path, false);

            std::cout << "Tokens: " << input_ids.size() << std::endl;
            std::cout << "Running benchmark (1000 iterations)..." << std::endl;

            // Warmup
            for (int i = 0; i < 50; ++i) embedder.embed(input_ids, attention_mask);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 1000; ++i) {
                embedder.embed(input_ids, attention_mask);
            }
            auto end = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            double avg_ms = total_ms / 1000.0;

            auto embedding = embedder.embed(input_ids, attention_mask);
            std::cout << "\nEmbedding dim: " << embedding.size() << std::endl;
            std::cout << "==================================================" << std::endl;
            std::cout << "PURE INFERENCE LATENCY: " << avg_ms << " ms" << std::endl;
            std::cout << "==================================================" << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
