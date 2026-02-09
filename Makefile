# Makefile for Arctic Embed - LibTorch + MPS (Apple Silicon GPU)
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -Ofast -flto -ffast-math -DNDEBUG

# Homebrew PyTorch paths
TORCH_DIR = /opt/homebrew/Cellar/pytorch/2.10.0
TORCH_SITE = $(TORCH_DIR)/libexec/lib/python3.14/site-packages/torch

INCLUDES = -I$(TORCH_SITE)/include \
           -I$(TORCH_SITE)/include/torch/csrc/api/include \
           -I$(TORCH_DIR)/include

LDFLAGS = -L$(TORCH_SITE)/lib \
          -L$(TORCH_DIR)/lib \
          -ltorch -ltorch_cpu -lc10 \
          -framework Foundation -framework Accelerate \
          -flto \
          -Wl,-rpath,$(TORCH_SITE)/lib \
          -Wl,-rpath,$(TORCH_DIR)/lib

SRC = src/arctic_embed_libtorch.cpp
TARGET = bin/arctic_embed_libtorch

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Build complete: $@"
	@ls -lh $@

clean:
	rm -f $(TARGET)

test: $(TARGET)
	PYTORCH_ENABLE_MPS_FALLBACK=1 ./$(TARGET) arctic_model_mps.pt "Hello, OpenClaw!"

.PHONY: all clean test
