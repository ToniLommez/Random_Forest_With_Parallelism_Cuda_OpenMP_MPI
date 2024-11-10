#ifndef CART_H
#define CART_H

#include "../include/node.hpp"
#include "../include/utils.hpp"
#include <string>
#include <vector>

using namespace std;

class Cart {
  public:
    Cart(const int max_depth, const int min_samples_split, const bool isClassification);
    Cart();
    ~Cart();

    void fit(float_matrix &X_train, float_vector &y_train);
    float_vector predict(float_matrix &X_test);

  private:
    Node *root;
    bool isClassification;

    // Train
    float_matrix X_train;
    float_vector y_train;

    Node *split_node(const float_matrix &X, const float_vector &y, const int depth);
    bool is_pure(const float_vector &y);
    float gini(const float_vector &y);
    pair<int, float> best_threshold(const float_matrix &X, const float_vector &y);
    tuple<float_matrix, float_vector, float_matrix, float_vector> divide(const float_matrix &X, const float_vector &y, int feature, float threshold);
    Node *new_leaf(const float_vector &y);

    // Prediction
    int n_features;
    int max_depth;
    size_t min_samples_split;

    float predict_single(const std::vector<float> &sample, Node *node);
};

#endif
