OS := $(shell uname)
CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra \
           -Wno-unused-variable -Wno-unused-parameter \
           -Wno-unused-private-field
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
CSV_PATH =
NoP =

# Verificar se é para utilizar OpenMP
ifeq ($(MAKECMDGOALS), omp)
    CXXFLAGS += -DOMP -fopenmp
endif

# Verificar se é para utilizar MPI
ifeq ($(MAKECMDGOALS), mpi)
    CXX = mpic++
    CXXFLAGS += -DENABLE_MPI -fopenmp
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

mpi: re

build: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJS)
	@$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -c main.cpp -o $@

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

run_mpi: build
	@mpirun -np $(NoP)  $(TARGET) -csv $(CSV_PATH)

exec:
	@$(MAKE) run CSV_PATH=$(CSV)

exec_mpi:
	@$(MAKE) run_mpi CSV_PATH=$(CSV) NoP=$(NP)

.PHONY: all build clean_obj clean_bin clean run exec re omp mpi run_mpi exec_mpi