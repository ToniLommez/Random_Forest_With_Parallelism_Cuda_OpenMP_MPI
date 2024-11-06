OS := $(shell uname)
CXX = clang++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -Werror \
           -Wno-unused-variable -Wno-unused-parameter \
           -Wno-unused-private-field \
           -stdlib=libc++ -I/usr/include/c++/v1
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
CSV_PATH = 

ifeq ($(MAKECMDGOALS), omp)
	CXXFLAGS += -DOMP -fopenmp
endif

SRCS = $(wildcard $(SRC_DIR)/*.cpp) main.cpp 
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

ifeq ($(OS), Windows_NT)
	RM = powershell -Command "Remove-Item -Recurse -Force"
	MKDIR = mkdir
	TARGET = $(BIN_DIR)/main.exe
else
	RM = rm -rf
	MKDIR = mkdir -p
	TARGET = $(BIN_DIR)/main
endif

all: build

re: clean all

omp: re

build: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

re: clean all

$(TARGET): $(OBJS)
	@ $(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ $(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	@ $(CXX) $(CXXFLAGS) -c main.cpp -o $@

$(BIN_DIR):
	@ $(MKDIR) $(BIN_DIR)

$(OBJ_DIR):
	@ $(MKDIR) $(OBJ_DIR)

clean_obj:
	@ $(RM) $(OBJ_DIR)/*

clean_bin:
	@ $(RM) $(BIN_DIR)/*

clean: clean_obj clean_bin

run: clean_all build clean_obj
	@ $(TARGET) -csv $(CSV_PATH)

exec:
	@ $(MAKE) run CSV_PATH=$(CSV)

.PHONY: all build clean_obj clean_bin clean_all run exec re omp
