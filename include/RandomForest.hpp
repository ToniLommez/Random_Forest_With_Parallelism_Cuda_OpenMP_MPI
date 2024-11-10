#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "../include/cart.hpp"
#include "../include/utils.hpp"

using namespace std;

class Random_Forest {
  public:
    Random_Forest();
    Random_Forest(int n_trees, int max_depth, int min_samples_split, bool isClassification);
    ~Random_Forest();

    void fit(float_matrix &X_train, float_vector &y_train);
    float_vector predict(float_matrix &X_test);

  private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    bool isClassification;

    Cart* root;
    vector<Cart*> trees;

#ifdef ENABLE_MPI
    float_cube trees_X_train;
    float_matrix trees_y_train;
    float_matrix trees_preds;
#endif

    pair<float_matrix, float_vector> bootstrap_sample(const float_matrix &X, const float_vector &y);
};

#endif // RANDOM_FOREST_H
