#include "../include/utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cstring>

using namespace std;

void trainTestSplit(const float_matrix& X, const float_vector& y,
                    float_matrix& X_train, float_vector& y_train,
                    float_matrix& X_test, float_vector& y_test,
                    int random_state, float test_size = 0.2) {
    int total_size = X.size();
    int test_size_count = static_cast<int>(total_size * test_size);

    // Random indexes for division
    vector<int> indices(total_size);
    iota(indices.begin(), indices.end(), 0);
    default_random_engine rng(random_state);
    shuffle(indices.begin(), indices.end(), rng);

    // Split data
    for (int i = 0; i < total_size; ++i) {
        if (i < test_size_count) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        } else {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
}

float calculateAccuracy(const float_vector& y_true, const float_vector& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / y_true.size();
}

#ifdef ENABLE_MPI

void serialize_vector(vector<char>& buf, vector<float>& vec) {
    size_t size = vec.size();
    char* ptr = reinterpret_cast<char*>(&size);
    buf.insert(buf.end(), ptr, ptr + sizeof(size_t));
    buf.insert(buf.end(), reinterpret_cast<char*>(vec.data()), reinterpret_cast<char*>(vec.data()) + vec.size() * sizeof(float));
}

void serialize_matrix(vector<char>& buf, vector<vector<float>>& vec) {
    size_t size = vec.size();
    char* ptr = reinterpret_cast<char*>(&size);
    buf.insert(buf.end(), ptr, ptr + sizeof(size_t));
    for(size_t i = 0; i < size; i++) {
        serialize_vector(buf, vec[i]);
    }
}

void deserialize_vector(vector<char>& buf, vector<float>& vec, size_t& idx) {
    size_t size;
    memcpy(&size, &buf[idx], sizeof(size_t));
    idx += sizeof(size_t);
    vec.resize(size);

    memcpy(vec.data(), &buf[idx], size * sizeof(float));
    idx += size * sizeof(float);
}

void deserialize_matrix(vector<char>& buf, vector<vector<float>>& vec, size_t& idx) {
    size_t matrix_size;
    memcpy(&matrix_size, &buf[idx], sizeof(size_t));
    idx += sizeof(size_t);
    vec.resize(matrix_size);
    
    for(size_t i = 0; i < matrix_size; i++) {
        size_t row_size;
        memcpy(&row_size, &buf[idx], sizeof(size_t));
        idx += sizeof(size_t);
        vec[i].resize(row_size);

        memcpy(vec[i].data(), &buf[idx], row_size * sizeof(float));
        idx += row_size * sizeof(float);
    }
}

#endif
