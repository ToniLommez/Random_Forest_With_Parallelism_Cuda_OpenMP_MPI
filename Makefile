OS := $(shell uname)
CXX = g++
CUDA = nvcc
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra \
           -Wno-unused-variable -Wno-unused-parameter \
           -Wno-unused-private-field \
		   -Wno-cast-function-type
CUDAFLAGS = -std=c++17 -Iinclude -arch=sm_60 -DCUDA
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
CSV_PATH =
NoP =

# Verificar se é para utilizar OpenMP
ifeq ($(MAKECMDGOALS), omp)
    CXXFLAGS += -DOMP -fopenmp
endif

ifeq ($(MAKECMDGOALS), cuda)
	CXXFLAGS += -DCUDA
endif

# Source files and object files
SRCS = $(wildcard $(SRC_DIR)/*.cpp) main.cpp
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)  # CUDA source files
CPP_OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJS = $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

ifeq ($(OS), Windows_NT)
    RM = powershell -Command "Remove-Item -Recurse -Force"
    MKDIR = mkdir
    TARGET = $(BIN_DIR)/main.exe
    CUDA_TARGET = $(BIN_DIR)/cuda_main.exe
else
    RM = rm -rf
    MKDIR = mkdir -p
    TARGET = $(BIN_DIR)/main
    CUDA_TARGET = $(BIN_DIR)/cuda_main
endif

all: build

re: clean all

omp: re

build: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

# Original target: compile and link C++ files only
$(TARGET): $(CPP_OBJS)
	@$(CXX) $(CXXFLAGS) -o $@ $^

# CUDA-specific target: compile and link C++ and CUDA files
$(CUDA_TARGET): $(CPP_OBJS) $(CUDA_OBJS)
	@$(CUDA) $(CUDAFLAGS) -o $@ $^

# Compile C++ source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@$(CUDA) $(CUDAFLAGS) -c $< -o $@

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

$(OBJ_DIR):
	@$(MKDIR) $(OBJ_DIR)

clean_obj:
	@$(RM) $(OBJ_DIR)/*

clean_bin:
	@$(RM) $(BIN_DIR)/*

clean: clean_obj clean_bin

run: build
	@$(TARGET) -csv $(CSV_PATH)

exec:
	@$(MAKE) run CSV_PATH=$(CSV)

cuda: clean $(CUDA_TARGET)

.PHONY: all build clean_obj clean_bin clean run exec re omp cuda

