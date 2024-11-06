CXX = clang++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -Werror
CXXFLAGS += -Wno-unused-variable -Wno-unused-parameter
CXXFLAGS += -Wno-unused-private-field
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
TARGET = $(BIN_DIR)/random_forest.exe

SRCS = $(wildcard $(SRC_DIR)/*.cpp) main.cpp 
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: build

build: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJS)
	@ $(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ $(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	@ $(CXX) $(CXXFLAGS) -c main.cpp -o $@

$(BIN_DIR):
	@ mkdir $(BIN_DIR)

$(OBJ_DIR):
	@ mkdir $(OBJ_DIR)

clean_obj:
	@ powershell -Command "Remove-Item -Recurse -Force $(OBJ_DIR)/*"

clean_bin:
	@ powershell -Command "Remove-Item -Recurse -Force $(BIN_DIR)/*"

clean_all: clean_obj clean_bin

run: clean_all build clean_obj
	@ $(TARGET)

.PHONY: all build clean_obj clean_bin clean_all run
