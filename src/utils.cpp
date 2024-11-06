#include "../include/utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

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
