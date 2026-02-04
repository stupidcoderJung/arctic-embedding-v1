# Makefile for Arctic Embed Tiny C++ Application
# Optimized for MacBook Air M1 with 8GB RAM

# Compiler settings
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -arch arm64
LDFLAGS = -framework Foundation -framework Accelerate

# ONNX Runtime settings
# You may need to adjust these paths based on your ONNX Runtime installation
ONNXRUNTIME_DIR = /usr/local/lib/libonnxruntime.dylib
ONNXRUNTIME_INCLUDE = /usr/local/include/onnxruntime

# If ONNX Runtime is installed via vcpkg or other package managers, adjust paths accordingly
# For example, if using vcpkg:
# ONNXRUNTIME_DIR = ${VCPKG_ROOT}/installed/arm64-osx/lib/libonnxruntime.a
# ONNXRUNTIME_INCLUDE = ${VCPKG_ROOT}/installed/arm64-osx/include

# If ONNX Runtime is not installed, you can download it from:
# https://github.com/microsoft/onnxruntime/releases

# Define source and object files
SRC = arctic_embed_tiny.cpp
TARGET = arctic_embed_tiny

# Check if ONNX Runtime is installed in standard location
ONNXRUNTIME_EXISTS := $(shell test -f $(ONNXRUNTIME_DIR) && echo "yes")

ifeq ($(ONNXRUNTIME_EXISTS), yes)
    # Standard installation
    LIBS = -lonnxruntime
    INCLUDES = -I$(ONNXRUNTIME_INCLUDE)
else
    # Look for ONNX Runtime in common locations
    ifneq ("$(wildcard /opt/homebrew/lib/libonnxruntime.dylib)", "")
        # Homebrew installation on Apple Silicon
        ONNXRUNTIME_DIR = /opt/homebrew/lib/libonnxruntime.dylib
        ONNXRUNTIME_INCLUDE = /opt/homebrew/include/onnxruntime
        LIBS = -lonnxruntime
        INCLUDES = -I$(ONNXRUNTIME_INCLUDE)
    else ifneq ("$(wildcard ./onnxruntime-osx-arm64/lib/libonnxruntime.dylib)", "")
        # Local ONNX Runtime installation
        ONNXRUNTIME_DIR = ./onnxruntime-osx-arm64/lib/libonnxruntime.dylib
        ONNXRUNTIME_INCLUDE = ./onnxruntime-osx-arm64/include
        LIBS = -L./onnxruntime-osx-arm64/lib -lonnxruntime
        INCLUDES = -I$(ONNXRUNTIME_INCLUDE)
    else
        # If not found, provide instructions
        $(error ONNX Runtime library not found. Please install ONNX Runtime for macOS ARM64.)
    endif
endif

# Optimization flags for M1
CXXFLAGS += -mcpu=apple-m1 -mtune=apple-m1

# Linker flags
LDFLAGS += -L$(dir $(ONNXRUNTIME_DIR))

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Install ONNX Runtime if needed
install-onnxruntime:
	@echo "Installing ONNX Runtime for macOS ARM64..."
	@mkdir -p ./onnxruntime-osx-arm64
	@curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz | tar xz -C ./onnxruntime-osx-arm64
	@echo "ONNX Runtime installed in ./onnxruntime-osx-arm64"

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Very clean (removes ONNX Runtime installation too)
distclean: clean
	rm -rf ./onnxruntime-osx-arm64

# Run the application (example)
run: $(TARGET)
	@echo "Running Arctic Embed Tiny..."
	@./$(TARGET) ./model.onnx "Hello, world!"

# Help target
help:
	@echo "Available targets:"
	@echo "  all               - Build the application"
	@echo "  install-onnxruntime - Download and install ONNX Runtime for macOS ARM64"
	@echo "  clean             - Remove build artifacts"
	@echo "  distclean         - Remove build artifacts and ONNX Runtime installation"
	@echo "  run               - Run the application with example input"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Note: You need to provide the path to the Arctic Embed Tiny ONNX model."
	@echo "Usage: ./$(TARGET) <model_path> \"<input_text>\""

.PHONY: all clean distclean run help install-onnxruntime