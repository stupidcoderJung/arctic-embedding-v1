#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <fstream>

class ArcticEmbedTiny {
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ArcticEmbedTiny"};
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};  // REUSE!

    // Model metadata
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_;
    std::vector<std::vector<int64_t>> output_node_dims_;

    // Tokenizer
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t unk_token_id_{100};
    int64_t cls_token_id_{101};
    int64_t sep_token_id_{102};
    int64_t pad_token_id_{0};
    size_t max_length_{512};

    // Reusable buffers
    std::vector<int64_t> input_ids_buffer_;
    std::vector<int64_t> attention_mask_buffer_;
    std::vector<int64_t> token_type_ids_buffer_;
    std::vector<float> pooled_result_buffer_;
    std::vector<float> embedding_buffer_;

public:
    ArcticEmbedTiny(const std::string& model_path, const std::string& vocab_path = "") {
        // Configure session options for MAXIMUM performance
        Ort::SessionOptions session_options;

        // CRITICAL FIXES:
        session_options.SetIntraOpNumThreads(4);  // M1 has 4 performance cores
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableMemPattern();  // FIX #1: ENABLE (not disable!) memory pattern optimization
        session_options.EnableCpuMemArena();

        // Create session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        // Get input and output info
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();

        input_node_names_.resize(num_input_nodes);
        output_node_names_.resize(num_output_nodes);
        input_node_dims_.resize(num_input_nodes);
        output_node_dims_.resize(num_output_nodes);

        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(i, allocator_);
            input_node_names_[i] = std::string(input_name_ptr.get());

            auto input_typeinfo = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_typeinfo.GetTensorTypeAndShapeInfo();
            input_node_dims_[i] = input_tensor_info.GetShape();
        }

        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
            output_node_names_[i] = std::string(output_name_ptr.get());

            auto output_typeinfo = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_typeinfo.GetTensorTypeAndShapeInfo();
            output_node_dims_[i] = output_tensor_info.GetShape();
        }

        // Initialize vocabulary
        if (!vocab_path.empty()) {
            initialize_vocab(vocab_path);
        } else {
            initialize_builtin_vocab();
        }

        // Pre-allocate buffers
        input_ids_buffer_.reserve(max_length_);
        attention_mask_buffer_.reserve(max_length_);
        token_type_ids_buffer_.reserve(max_length_);
        pooled_result_buffer_.resize(384);
        embedding_buffer_.resize(384);
    }

    std::vector<float> embed(const std::string& text, bool silent = false) {
        // Tokenize
        tokenize_text_into_buffer(text);

        // Determine input shape
        std::vector<int64_t> input_shape = fix_tensor_shape(input_node_dims_[0], input_ids_buffer_.size());

        // Create input tensors (reusing buffers)
        std::vector<Ort::Value> input_tensors;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            input_ids_buffer_.data(),
            input_ids_buffer_.size(),
            input_shape.data(),
            input_shape.size()
        ));

        if (input_node_names_.size() >= 2) {
            std::vector<int64_t> mask_shape = fix_tensor_shape(input_node_dims_[1], attention_mask_buffer_.size());
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_,
                attention_mask_buffer_.data(),
                attention_mask_buffer_.size(),
                mask_shape.data(),
                mask_shape.size()
            ));
        }

        if (input_node_names_.size() >= 3) {
            std::vector<int64_t> token_type_shape = fix_tensor_shape(input_node_dims_[2], token_type_ids_buffer_.size());
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_,
                token_type_ids_buffer_.data(),
                token_type_ids_buffer_.size(),
                token_type_shape.data(),
                token_type_shape.size()
            ));
        }

        // Run inference
        std::vector<const char*> input_names;
        for(const auto& name : input_node_names_) input_names.push_back(name.c_str());

        std::vector<const char*> output_names;
        for(const auto& name : output_node_names_) output_names.push_back(name.c_str());

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        // Extract output
        float* float_array_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTypeInfo();
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = tensor_info.GetShape();

        int64_t hidden_size = output_shape[2];
        int actual_seq_len = std::min(static_cast<int>(output_shape[1]), static_cast<int>(input_ids_buffer_.size()));

        // Mean pooling (reusing pooled_result_buffer_)
        std::fill(pooled_result_buffer_.begin(), pooled_result_buffer_.end(), 0.0f);
        int valid_tokens_count = 0;

        for (int i = 0; i < actual_seq_len; ++i) {
            if (attention_mask_buffer_[i] == 1) {
                valid_tokens_count++;
            }
        }

        for (int h = 0; h < hidden_size; ++h) {
            float sum = 0.0f;
            for (int s = 0; s < actual_seq_len; ++s) {
                if (attention_mask_buffer_[s] == 1) {
                    sum += float_array_data[s * hidden_size + h];
                }
            }
            pooled_result_buffer_[h] = (valid_tokens_count > 0) ? (sum / valid_tokens_count) : 0.0f;
        }

        // Copy to embedding_buffer and normalize
        std::copy(pooled_result_buffer_.begin(), pooled_result_buffer_.begin() + 384, embedding_buffer_.begin());
        normalize_embedding(embedding_buffer_);

        // FIX #2: Only output if not silent (to avoid I/O overhead during benchmarking)
        if (!silent) {
            std::cout << "[";
            for (size_t i = 0; i < embedding_buffer_.size(); ++i) {
                std::cout << embedding_buffer_[i];
                if (i != embedding_buffer_.size() - 1) {
                    std::cout << ",";
                }
            }
            std::cout << "]" << std::endl;
            std::cout << "Generated embedding of size: " << embedding_buffer_.size() << std::endl;
        }

        return embedding_buffer_;
    }

private:
    void initialize_builtin_vocab() {
        vocab_["[UNK]"] = unk_token_id_;
        vocab_["[CLS]"] = cls_token_id_;
        vocab_["[SEP]"] = sep_token_id_;
        vocab_["[PAD]"] = pad_token_id_;
        vocab_["the"] = 1000;
        vocab_["a"] = 1001;
        vocab_["openclaw"] = 1002;
        vocab_["is"] = 1049;
        vocab_["an"] = 1002;
        vocab_["ai"] = 2000;
        vocab_["assistant"] = 2001;
        vocab_["framework"] = 2002;
    }

    bool initialize_vocab(const std::string& vocab_path) {
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            initialize_builtin_vocab();
            return false;
        }

        vocab_.clear();
        std::string token;
        int64_t id = 0;

        while (std::getline(vocab_file, token)) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            vocab_[token] = id++;
        }

        vocab_file.close();
        return true;
    }

    void tokenize_text_into_buffer(const std::string& text) {
        input_ids_buffer_.clear();
        attention_mask_buffer_.clear();
        token_type_ids_buffer_.clear();

        // Simple tokenization
        std::istringstream iss(text);
        std::string token;

        std::vector<int64_t> tokens;
        while (iss >> token) {
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unk_token_id_);
            }
        }

        // Truncate if needed
        if (tokens.size() > max_length_ - 2) {
            tokens.resize(max_length_ - 2);
        }

        // Add special tokens
        input_ids_buffer_.push_back(cls_token_id_);
        for (auto t : tokens) {
            input_ids_buffer_.push_back(t);
        }
        input_ids_buffer_.push_back(sep_token_id_);

        // Create attention mask and token type ids
        attention_mask_buffer_.resize(input_ids_buffer_.size(), 1);
        token_type_ids_buffer_.resize(input_ids_buffer_.size(), 0);
    }

    std::vector<int64_t> fix_tensor_shape(const std::vector<int64_t>& original_shape, size_t sequence_length) {
        std::vector<int64_t> fixed_shape = original_shape;
        for (size_t i = 0; i < fixed_shape.size(); ++i) {
            if (fixed_shape[i] <= 0) {
                fixed_shape[i] = (i == 1) ? static_cast<int64_t>(sequence_length) : 1;
            }
        }
        return fixed_shape;
    }

    void normalize_embedding(std::vector<float>& embedding) {
        float sum = 0.0f;
        for (float val : embedding) {
            sum += val * val;
        }
        sum = std::sqrt(sum);

        if (sum > 0.0f) {
            for (float& val : embedding) {
                val /= sum;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_text> [vocab_path]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_text = argv[2];
    std::string vocab_path = (argc == 4) ? argv[3] : "";

    try {
        std::unique_ptr<ArcticEmbedTiny> embedder = std::make_unique<ArcticEmbedTiny>(model_path, vocab_path);
        std::vector<float> embedding = embedder->embed(input_text, false);  // silent = false for display

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
