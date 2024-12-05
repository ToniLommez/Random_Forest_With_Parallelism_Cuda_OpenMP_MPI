#include "../include/RandomForest.hpp"
#include "../include/cart.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>

#ifdef OMP
#include <omp.h>
#endif

/**
 * Sequencial
 * real    3m50,563s
 * user    3m50,425s
 * sys     0m0,016s
 * 
 *
 * OMP GPU only
 * pragma omp target teams distribute parallel for
 * real    1m16,720s
 * user    7m22,167s
 * sys     0m0,217s
 *
 *
 * OMP GPU and OMP CPU
 * pragma omp target teams distribute parallel for schedule(dynamic)
 * 
 * omp_set_num_threads(2);
 * pragma omp parallel for schedule(dynamic)
 * real    1m13,476s
 * user    7m24,257s
 * sys     0m0,140s
 *
 *
 * OMP GPU with num_teams and OMP CPU
 * pragma omp target teams distribute parallel for num_teams(4)
 * 
 * omp_set_num_threads(4);
 * pragma omp parallel for schedule(dynamic)
 * real    1m13,550s
 * user    7m12,704s
 * sys     0m0,144s
 * 
 * 
 * OMP GPU with num_teams and OMP CPU
 * pragma omp target teams distribute parallel for num_teams(2)
 * 
 * omp_set_num_threads(4);
 * pragma omp parallel for schedule(dynamic)
 * real    1m12,864s
 * user    7m15,514s
 * sys     0m0,096s
 * 
 * 
 * OMP GPU with num_teams and OMP CPU
 * pragma omp target teams distribute parallel for num_teams(1)
 * 
 * omp_set_num_threads(4);
 * pragma omp parallel for schedule(dynamic)
 * real    1m13,799s
 * user    7m14,568s
 * sys     0m0,492s
 */

using namespace std;

// Construtor
Random_Forest::Random_Forest(int n_trees, int max_depth, int min_samples_split, bool isClassification)
    : n_trees(n_trees), max_depth(max_depth), min_samples_split(min_samples_split), isClassification(isClassification) {
    for(int i = 0; i < n_trees; i++) {
        trees.push_back(new Cart(max_depth, min_samples_split, isClassification));
    }
}

Random_Forest::Random_Forest()
    : n_trees(100), max_depth(15), min_samples_split(2), isClassification(true) {
}

// Destrutor
Random_Forest::~Random_Forest() {
    for (Cart *tree : trees) {
        delete tree;
    }
}

// Função de amostragem com reposição (bootstrap)
pair<float_matrix, float_vector> Random_Forest::bootstrap_sample(const float_matrix &X, const float_vector &y) {
    float_matrix X_sample;
    float_vector y_sample;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, X.size() - 1);

    for(size_t i = 0; i < X.size(); ++i) {
        int index = dis(gen);
        X_sample.push_back(X[index]);
        y_sample.push_back(y[index]);
    }
    return {X_sample, y_sample};
}

// Train
void Random_Forest::fit(float_matrix &X_train, float_vector &y_train) {
#ifdef OMP
    #pragma omp target teams distribute parallel for num_teams(8) schedule(dynamic)
#endif
    for (Cart* tree : trees) {
        auto [X_sample, y_sample] = bootstrap_sample(X_train, y_train);
        tree->fit(X_sample, y_sample);
    }
}

// Predict
float_vector Random_Forest::predict(float_matrix &X_test) {
    vector<float_vector> all_predictions;

    for(Cart* tree : trees) {
        all_predictions.push_back(tree->predict(X_test));
    }

    float_vector predictions;
    for(size_t i = 0; i < X_test.size(); ++i) {
        unordered_map<float, int> class_counts;

        for(const auto &preds : all_predictions) {
            class_counts[preds[i]]++;
        }

        // Encontrar a previsão mais comum (voto majoritário)
        float majority_class = max_element(
            class_counts.begin(), 
            class_counts.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; }
        )->first;
        predictions.push_back(majority_class);
    }

    return predictions;
}
