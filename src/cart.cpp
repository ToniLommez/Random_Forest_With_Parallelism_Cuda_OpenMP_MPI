#include "../include/cart.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <cstring>
#include <iostream>

#ifdef OMP
#include <omp.h>
#endif

#ifdef CUDA
#include "../include/tree.hpp"
#endif

using namespace std;

Cart::Cart() {
    n_features = 0;
    max_depth = 15;
    min_samples_split = 10;
    isClassification = true;
    root = nullptr;
}

Cart::Cart(const int max_depth, const int min_samples_split, const bool isClassification)
    : isClassification(isClassification), max_depth(max_depth), min_samples_split(min_samples_split) {}

Cart::~Cart() {
    delete root;
}

void Cart::fit(float_matrix &X_train, float_vector &y_train) {
    this->X_train = X_train;
    this->y_train = y_train;
    this->n_features = X_train[0].size();
    this->root = split_node(X_train, y_train, 0);
}

Node *Cart::split_node(const float_matrix &X, const float_vector &y, const int depth) {
    // Stop condition
    if (depth >= max_depth || y.size() < min_samples_split || is_pure(y)) {
        return new_leaf(y);
    }

    // Best division
    auto [feature, threshold] = best_threshold(X, y);

    // Divide data
    auto [X_left, y_left, X_right, y_right] = divide(X, y, feature, threshold);

    // Create nodes
    Node *node = new Node(feature, false, threshold, 0);
    node->left = split_node(X_left, y_left, depth + 1);
    node->right = split_node(X_right, y_right, depth + 1);

    return node;
}

// isPure test if all values present are from the same type
bool Cart::is_pure(const float_vector &y) {
    if (y.empty()) return true;

    for (size_t i = 1; i < y.size(); ++i) {
        if (y[i] != y[0]) {
            return false;
        }
    }
    return true;
}

float Cart::gini(const float_vector &y) {
    // number of occurrences per class
    unordered_map<float, int> class_count;
    for (float label : y) {
        class_count[label]++;
    }

    // calculate gini
    float gini = 1.0;
    int total = y.size();
    for (const auto &[classe, count] : class_count) {
        float prob = static_cast<float>(count) / total;
        gini -= prob * prob;
    }
    return gini;
}

pair<int, float> Cart::best_threshold(const float_matrix &X, const float_vector &y) {
    int best_feature = -1;
    float best_threshold = 0;
    float lowest_impurity = numeric_limits<float>::max();

#ifdef CUDA
    // std::cout << "X[0].size(): " << X[0].size() << std::endl;
    cuda_best_threshold_sender(X, y, X.size(), X[0].size(), &best_feature, &best_threshold);
#else
    // For every feature find the best threshold
#ifdef OMP
    omp_set_num_threads(64);
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t feature = 0; feature < X[0].size(); ++feature) {
        for (const auto &samples : X) {
            float threshold = samples[feature];

            auto [X_left, y_left, X_right, y_right] = divide(X, y, feature, threshold);

            // Calculate gini value for every group
            float gini_left = gini(y_left);
            float gini_right = gini(y_right);
            float weighted_impurity = (y_left.size() * gini_left + y_right.size() * gini_right) / y.size();

            // is it the lowest impurity?

#ifdef OMP
            #pragma omp critical
            {
#endif
                if (weighted_impurity < lowest_impurity) {
                    lowest_impurity = weighted_impurity;
                    best_feature = feature;
                    best_threshold = threshold;
                }
#ifdef OMP
            }
#endif
        }
    }
#endif
    return {best_feature, best_threshold};
}

tuple<float_matrix, float_vector, float_matrix, float_vector> Cart::divide(const float_matrix &X, const float_vector &y, int feature, float threshold) {
    float_matrix X_left, X_right;
    float_vector y_left, y_right;

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature] <= threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    return {X_left, y_left, X_right, y_right};
}

Node *Cart::new_leaf(const float_vector &y) {
    Node *leaf = new Node();
    leaf->is_leaf = true;

    if (isClassification) {
        std::unordered_map<float, int> classCounts;
        for (float label : y) {
            classCounts[label]++;
        }
        leaf->prediction = max_element(classCounts.begin(), classCounts.end(), [](const auto &a, const auto &b) { return a.second < b.second; })->first;
    } else {
        float sum = accumulate(y.begin(), y.end(), 0.0f);
        leaf->prediction = sum / y.size();
    }

    return leaf;
}

float_vector Cart::predict(float_matrix& X_test) {
    float_vector predictions;

    for (const auto& sample : X_test) {
        predictions.push_back(predict_single(sample, root));
    }

    return predictions;
}


float Cart::predict_single(const std::vector<float>& sample, Node* node) {
    if (node->is_leaf) {
        return node->prediction;
    }

    if (sample[node->feature] <= node->threshold) {
        return predict_single(sample, node->left);
    } else {
        return predict_single(sample, node->right);
    }
}
