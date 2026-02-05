// Arctic Embed Tiny - LibTorch Implementation
// Uses PyTorch C++ API for same performance as Python
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

class ArcticEmbedLibTorch {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    
public:
    ArcticEmbedLibTorch(const std::string& model_path) 
        : device_(torch::kMPS) {  // FORCE GPU 가속!
        
        std::cout << "🚀 Using Metal Performance Shaders (MPS) for GPU acceleration" << std::endl;
        
        // Load TorchScript model
        try {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
            std::cout << "✅ Model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "❌ Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    std::vector<float> embed(const std::vector<int64_t>& input_ids,
                             const std::vector<int64_t>& attention_mask) {
        torch::NoGradGuard no_grad;
        
        // Convert to tensors on CPU first, then move to target device
        auto ids_tensor = torch::from_blob(
            const_cast<int64_t*>(input_ids.data()),
            {1, static_cast<int64_t>(input_ids.size())},
            torch::kLong
        ).clone().to(device_);  // Clone to avoid modifying original data
        
        auto mask_tensor = torch::from_blob(
            const_cast<int64_t*>(attention_mask.data()),
            {1, static_cast<int64_t>(attention_mask.size())},
            torch::kLong
        ).clone().to(device_);
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(ids_tensor);
        inputs.push_back(mask_tensor);
        
        auto output_dict = model_.forward(inputs).toGenericDict();
        auto last_hidden_state = output_dict.at("last_hidden_state").toTensor();
        
        // Mean pooling
        auto pooled = last_hidden_state.mean(1).squeeze(0);
        
        // Normalize
        auto norm = pooled.norm(2);
        auto normalized = pooled / norm;
        
        // Convert to CPU and std::vector
        auto cpu_tensor = normalized.to(torch::kCPU);
        auto data_ptr = cpu_tensor.data_ptr<float>();
        
        return std::vector<float>(data_ptr, data_ptr + cpu_tensor.numel());
    }
};

// Simple tokenizer (will be replaced with proper tokenizer later)
std::pair<std::vector<int64_t>, std::vector<int64_t>> simple_tokenize(const std::string& text) {
    // This is a placeholder - in production, use proper BERT tokenizer
    std::vector<int64_t> input_ids = {101, 1000, 2000, 3000, 102};  // [CLS] ... [SEP]
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    return {input_ids, attention_mask};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <torchscript_model_path> <input_text>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string input_text = argv[2];
    
    try {
        std::cout << "==================================================" << std::endl;
        std::cout << "Arctic Embed - LibTorch (PyTorch C++) Version" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << std::endl;
        
        {
            // Scope to ensure proper cleanup
            ArcticEmbedLibTorch embedder(model_path);
            
            auto [input_ids, attention_mask] = simple_tokenize(input_text);
            
            std::cout << "🚀 Running internal benchmark (1000 iterations)..." << std::endl;
            
            // Warmup (50 runs)
            for(int i=0; i<50; ++i) embedder.embed(input_ids, attention_mask);
            
            auto start = std::chrono::high_resolution_clock::now();
            for(int i=0; i<1000; ++i) {
                embedder.embed(input_ids, attention_mask);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            double avg_ms = total_ms / 1000.0;
            
            std::cout << "\n==================================================" << std::endl;
            std::cout << "🔥 PURE INFERENCE LATENCY: " << avg_ms << " ms" << std::endl;
            std::cout << "==================================================" << std::endl;
        }
        
        // Clean shutdown - let destructors run
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
