# Makefile for Arctic Embed (Optimized - 4 threads)
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native
LDFLAGS = -framework Foundation

ONNXRUNTIME_INCLUDE = /opt/homebrew/include/onnxruntime
LIBS = -L/opt/homebrew/lib -lonnxruntime
INCLUDES = -I$(ONNXRUNTIME_INCLUDE)

SRC = src/arctic_embed_tiny.cpp
TARGET = bin/arctic_embed_test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
