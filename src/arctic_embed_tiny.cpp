#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IOS || TARGET_OS_MAC
#define USE_COREML
#endif
#endif

#ifdef USE_COREML
#include <onnxruntime_c_api.h>
#ifdef COREML_PROVIDER_AVAILABLE
#include <coreml_provider_factory.h>
#else
#warning CoreML provider not available in this build of ONNX Runtime
#endif
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <sstream>

class ArcticEmbedTiny {
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ArcticEmbedTiny"};
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Model metadata
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_;
    std::vector<std::vector<int64_t>> output_node_dims_;

    // Tokenizer vocabulary mapping (simplified for demonstration)
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t unk_token_id_;
    int64_t cls_token_id_;
    int64_t sep_token_id_;
    int64_t pad_token_id_;
    size_t max_length_;

public:
    ArcticEmbedTiny(const std::string& model_path) : unk_token_id_(100), cls_token_id_(101), sep_token_id_(102), pad_token_id_(0), max_length_(512) {
        // Configure session options for optimal performance on M1
        Ort::SessionOptions session_options;

        // Optimize for CPU on M1
        session_options.SetIntraOpNumThreads(2);  // Reduce threads to save memory on 8GB systems
        session_options.SetInterOpNumThreads(2);  // Reduce threads to save memory on 8GB systems
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Disable memory pattern optimization to reduce memory usage
        session_options.DisableMemPattern();

#ifdef USE_COREML
#ifdef COREML_PROVIDER_AVAILABLE
        // Enable CoreML acceleration for M1 Mac
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0);
        if (status != nullptr) {
            const char* error_message;
            OrtGetErrorMessage(status, &error_message);
            std::cerr << "Warning: Failed to enable CoreML execution provider: " << (error_message ? error_message : "Unknown error") << std::endl;
            OrtReleaseStatus(status);
        } else {
            std::cout << "CoreML execution provider enabled successfully." << std::endl;
        }
#else
        std::cout << "CoreML provider not available in this build of ONNX Runtime." << std::endl;
#endif
#endif

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
            // Get input name - handle potential name variations
            Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(i, allocator_);
            const char* raw_name = input_name_ptr.get();

            // Store the actual name returned by the model
            input_node_names_[i] = std::string(raw_name);

            // Get input shape
            auto input_typeinfo = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_typeinfo.GetTensorTypeAndShapeInfo();
            input_node_dims_[i] = input_tensor_info.GetShape();

            // Print the actual input_node_dims_ values for debugging
            std::cout << "Input node " << i << " (" << input_node_names_[i] << ") dimensions: ";
            for(size_t j = 0; j < input_node_dims_[i].size(); j++) {
                std::cout << input_node_dims_[i][j];
                if(j < input_node_dims_[i].size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }

        for (size_t i = 0; i < num_output_nodes; i++) {
            // Get output name - handle potential name variations
            Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
            const char* raw_name = output_name_ptr.get();

            // Store the actual name returned by the model
            output_node_names_[i] = std::string(raw_name);

            // Get output shape
            auto output_typeinfo = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_typeinfo.GetTensorTypeAndShapeInfo();
            output_node_dims_[i] = output_tensor_info.GetShape();
        }

        // Initialize vocabulary for tokenization (in a real implementation, this would be loaded from a file)
        initialize_vocab();
    }

    ~ArcticEmbedTiny() {
        // Explicitly reset session to free memory
        session_.reset();
    }
    
    // Generate embeddings for input text
    std::vector<float> embed(const std::string& text) {
        std::vector<int64_t> input_ids = tokenize_text(text);
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0);

        // Determine input shape dynamically based on the actual model requirements
        std::vector<int64_t> input_shape;
        if (!input_node_dims_.empty() && !input_node_dims_[0].empty()) {
            // Use the model's expected input dimensions
            input_shape = input_node_dims_[0];
            // Update the sequence length dimension if it's variable (-1 or other placeholder)
            if (input_shape.size() >= 2 && (input_shape[1] == -1 || input_shape[1] <= 0)) {
                input_shape[1] = static_cast<int64_t>(input_ids.size());
            } else if (input_shape.size() == 1) {
                // If only batch size is specified, add sequence length
                input_shape.push_back(static_cast<int64_t>(input_ids.size()));
            }
        } else {
            // Fallback to default shape
            input_shape = {1, static_cast<int64_t>(input_ids.size())};
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensors dynamically based on model requirements
        std::vector<Ort::Value> input_tensors;

        // Check how many inputs the model expects and create appropriate tensors
        size_t num_inputs = input_node_names_.size();

        // Create input tensors based on available input types
        if (num_inputs >= 1) {
            // Fix the first input tensor shape to handle dynamic dimensions
            std::vector<int64_t> actual_input_shape = input_shape;
            if (!input_node_dims_[0].empty()) {
                actual_input_shape = fix_tensor_shape(input_node_dims_[0], input_ids.size());
            }

            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                input_ids.data(),
                input_ids.size(),
                actual_input_shape.data(),
                actual_input_shape.size()
            ));
        }

        if (num_inputs >= 2) {
            // Use the model's expected shape for attention mask, or fallback to input shape
            std::vector<int64_t> mask_shape;
            if (num_inputs >= 2 && !input_node_dims_[1].empty()) {
                mask_shape = fix_tensor_shape(input_node_dims_[1], attention_mask.size());
            } else {
                mask_shape = fix_tensor_shape(input_shape, attention_mask.size());
            }

            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                attention_mask.data(),
                attention_mask.size(),
                mask_shape.data(),
                mask_shape.size()
            ));
        }

        if (num_inputs >= 3) {
            // Use the model's expected shape for token type ids, or fallback to input shape
            std::vector<int64_t> token_type_shape;
            if (num_inputs >= 3 && !input_node_dims_[2].empty()) {
                token_type_shape = fix_tensor_shape(input_node_dims_[2], token_type_ids.size());
            } else {
                token_type_shape = fix_tensor_shape(input_shape, token_type_ids.size());
            }

            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                token_type_ids.data(),
                token_type_ids.size(),
                token_type_shape.data(),
                token_type_shape.size()
            ));
        }

        try {
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

            size_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }

            // Create result vector with proper size
            std::vector<float> result(output_size);
            std::copy(float_array_data, float_array_data + output_size, result.begin());

            // Normalize the embedding (L2 normalization)
            normalize_embedding(result);

            // Output only the vector as JSON for easy parsing
            std::cout << "[";
            for (size_t i = 0; i < result.size(); ++i) {
                std::cout << result[i];
                if (i != result.size() - 1) {
                    std::cout << ",";
                }
            }
            std::cout << "]" << std::endl;

            // Explicitly clear input tensors to free memory immediately
            input_tensors.clear();

            return result;

        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            return {};
        }
    }
    
private:
    // Initialize vocabulary for tokenization (in a real implementation, this would be loaded from a file)
    void initialize_vocab() {
        // This is a simplified vocabulary for demonstration purposes
        // In a real implementation, you would load this from a tokenizer file
        vocab_["[UNK]"] = unk_token_id_;
        vocab_["[CLS]"] = cls_token_id_;
        vocab_["[SEP]"] = sep_token_id_;
        vocab_["[PAD]"] = pad_token_id_;

        // Add some sample tokens (in a real implementation, this would be much larger)
        vocab_["the"] = 1000;
        vocab_["a"] = 1001;
        vocab_["an"] = 1002;
        vocab_["and"] = 1003;
        vocab_["or"] = 1004;
        vocab_["but"] = 1005;
        vocab_["in"] = 1006;
        vocab_["on"] = 1007;
        vocab_["at"] = 1008;
        vocab_["to"] = 1009;
        vocab_["for"] = 1010;
        vocab_["of"] = 1011;
        vocab_["with"] = 1012;
        vocab_["by"] = 1013;
        vocab_["from"] = 1014;
        vocab_["up"] = 1015;
        vocab_["about"] = 1016;
        vocab_["into"] = 1017;
        vocab_["through"] = 1018;
        vocab_["during"] = 1019;
        vocab_["before"] = 1020;
        vocab_["after"] = 1021;
        vocab_["above"] = 1022;
        vocab_["below"] = 1023;
        vocab_["between"] = 1024;
        vocab_["among"] = 1025;
        vocab_["he"] = 1026;
        vocab_["she"] = 1027;
        vocab_["it"] = 1028;
        vocab_["they"] = 1029;
        vocab_["we"] = 1030;
        vocab_["you"] = 1031;
        vocab_["i"] = 1032;
        vocab_["me"] = 1033;
        vocab_["him"] = 1034;
        vocab_["her"] = 1035;
        vocab_["us"] = 1036;
        vocab_["them"] = 1037;
        vocab_["my"] = 1038;
        vocab_["your"] = 1039;
        vocab_["his"] = 1040;
        vocab_["its"] = 1041;
        vocab_["our"] = 1042;
        vocab_["their"] = 1043;
        vocab_["this"] = 1044;
        vocab_["that"] = 1045;
        vocab_["these"] = 1046;
        vocab_["those"] = 1047;
        vocab_["am"] = 1048;
        vocab_["is"] = 1049;
        vocab_["are"] = 1050;
        vocab_["was"] = 1051;
        vocab_["were"] = 1052;
        vocab_["be"] = 1053;
        vocab_["been"] = 1054;
        vocab_["being"] = 1055;
        vocab_["have"] = 1056;
        vocab_["has"] = 1057;
        vocab_["had"] = 1058;
        vocab_["do"] = 1059;
        vocab_["does"] = 1060;
        vocab_["did"] = 1061;
        vocab_["will"] = 1062;
        vocab_["would"] = 1063;
        vocab_["could"] = 1064;
        vocab_["should"] = 1065;
        vocab_["may"] = 1066;
        vocab_["might"] = 1067;
        vocab_["must"] = 1068;
        vocab_["can"] = 1069;
        vocab_["shall"] = 1070;
    }

    // Simple whitespace-based tokenization (in a real implementation, you would use a proper tokenizer like WordPiece or BPE)
    std::vector<std::string> simple_tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;

        while (iss >> token) {
            // Convert to lowercase for basic case normalization
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
            tokens.push_back(token);
        }

        return tokens;
    }

    // Actual tokenizer - in practice, you would use the actual tokenizer
    std::vector<int64_t> tokenize_text(const std::string& text) {
        // This is a simplified tokenizer implementation
        // In reality, you would use the tokenizer associated with the Arctic model
        std::vector<int64_t> tokens;

        // Get tokens using simple whitespace tokenization
        std::vector<std::string> token_strings = simple_tokenize(text);

        // Convert tokens to IDs using vocabulary
        for (const std::string& token : token_strings) {
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // Handle unknown tokens
                tokens.push_back(unk_token_id_);
            }
        }

        // Truncate if too long
        if (tokens.size() > max_length_ - 2) {  // Reserve space for [CLS] and [SEP]
            tokens.resize(max_length_ - 2);
        }

        // Add special tokens if needed (CLS, SEP, etc.)
        tokens.insert(tokens.begin(), cls_token_id_); // [CLS] token
        tokens.push_back(sep_token_id_);              // [SEP] token

        // Note: We don't pad to max_length here because we want to preserve the actual sequence length
        // for dynamic shape handling. Padding will be handled by the model if needed internally.

        return tokens;
    }

    // Helper function to fix tensor shapes by replacing negative or zero dimensions with actual values
    std::vector<int64_t> fix_tensor_shape(const std::vector<int64_t>& original_shape, size_t sequence_length) {
        std::vector<int64_t> fixed_shape = original_shape;

        // Replace any negative or zero dimensions with appropriate values
        for (size_t i = 0; i < fixed_shape.size(); ++i) {
            if (fixed_shape[i] <= 0) {
                // For the sequence length dimension (usually index 1), use the actual sequence length
                if (i == 1) {
                    fixed_shape[i] = static_cast<int64_t>(sequence_length);
                } else {
                    // For other dimensions, use a default value of 1 if we can't determine the correct value
                    fixed_shape[i] = 1;
                }
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

// Utility function to measure execution time
template<typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Return milliseconds
}

// RAII wrapper for memory management
class MemoryManager {
public:
    static void force_garbage_collect() {
        // ONNX Runtime doesn't have a garbage collector, but we can ensure
        // all temporary tensors are released by calling sync functions
        // In practice, this means ensuring all Ort::Value objects go out of scope
    }

    // Memory monitoring for 8GB systems
    static void monitor_memory_usage() {
        // On Unix-like systems, we could check memory usage
        // For now, just a placeholder for memory monitoring
        #ifdef __linux__
        FILE* file = fopen("/proc/meminfo", "r");
        if (file) {
            char line[256];
            while (fgets(line, sizeof(line), file)) {
                if (strncmp(line, "MemAvailable:", 13) == 0) {
                    long mem_available_kb;
                    sscanf(line, "MemAvailable: %ld kB", &mem_available_kb);
                    double mem_available_gb = mem_available_kb / (1024.0 * 1024.0);
                    std::cout << "Available memory: " << mem_available_gb << " GB" << std::endl;
                    break;
                }
            }
            fclose(file);
        }
        #elif defined(__APPLE__)
        // On macOS, we could use system calls to check memory
        struct mach_task_basic_info info;
        mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
        kern_return_t kerr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size);
        if (kerr == KERN_SUCCESS) {
            double mem_used_gb = info.resident_size / (1024.0 * 1024.0 * 1024.0);
            std::cout << "Memory used by process: " << mem_used_gb << " GB" << std::endl;
        }
        #endif
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_text>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_text = argv[2];

    try {
        // Monitor memory before loading model
        std::cout << "Checking memory usage before model loading..." << std::endl;
        MemoryManager::monitor_memory_usage();

        // Create Arctic Embed Tiny instance
        std::cout << "Loading Arctic Embed Tiny model..." << std::endl;

        // Load model in a separate scope to ensure memory management
        std::unique_ptr<ArcticEmbedTiny> embedder = std::make_unique<ArcticEmbedTiny>(model_path);

        std::cout << "Generating embedding for: \"" << input_text << "\"" << std::endl;

        std::vector<float> embedding = embedder->embed(input_text);

        std::cout << "Generated embedding of size: " << embedding.size() << std::endl;

        // Explicitly clear the embedding vector to free memory
        std::vector<float>().swap(embedding);

        // Explicitly reset the embedder to release model memory immediately
        embedder.reset();

        std::cout << "Memory released after execution." << std::endl;

        // Monitor memory after execution
        std::cout << "Checking memory usage after execution..." << std::endl;
        MemoryManager::monitor_memory_usage();

        // Force cleanup of any remaining resources
        MemoryManager::force_garbage_collect();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}