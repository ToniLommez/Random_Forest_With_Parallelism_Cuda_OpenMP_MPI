#include "../include/RandomForest.hpp"
#include "../include/cart.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>

#ifdef ENABLE_MPI
#include <mpi.h>
#define MPI_N_PROCESSES 1
#endif

using namespace std;

// Construtor
Random_Forest::Random_Forest(int n_trees, int max_depth, int min_samples_split, bool isClassification)
    : n_trees(n_trees), max_depth(max_depth), min_samples_split(min_samples_split), isClassification(isClassification) {
#ifdef ENABLE_MPI
    int trees_per_process = n_trees / MPI_N_PROCESSES;
    int remaining_trees = n_trees % MPI_N_PROCESSES;
    int n_trees_for_main = trees_per_process + (remaining_trees > 0 ? 1 : 0);
    
    for(int i = 1; i < MPI_N_PROCESSES; ++i) {
        int n_trees_to_send = trees_per_process + (i <= remaining_trees ? 1 : 0);
        MPI_Send(&n_trees_to_send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        for(int j = 0; j < n_trees_to_send; j++) {
            MPI_Send(&max_depth, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&min_samples_split, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&isClassification, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }

    for(int i = 0; i < n_trees_for_main; i++) {
        trees.push_back(new Cart(max_depth, min_samples_split, isClassification));
    }
#else
    for(int i = 0; i < n_trees; i++) {
        trees.push_back(new Cart(max_depth, min_samples_split, isClassification));
    }
#endif
}

Random_Forest::Random_Forest()
    : n_trees(100), max_depth(15), min_samples_split(2), isClassification(true) {
}

// Destrutor
Random_Forest::~Random_Forest() {
#ifndef ENABLE_MPI
    for (Cart *tree : trees) {
        delete tree;
    }
#endif
}

// Função de amostragem com reposição (bootstrap)
pair<float_matrix, float_vector> Random_Forest::bootstrap_sample(const float_matrix &X, const float_vector &y) {
    float_matrix X_sample;
    float_vector y_sample;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, X.size() - 1);

    for (size_t i = 0; i < X.size(); ++i) {
        int index = dis(gen);
        X_sample.push_back(X[index]);
        y_sample.push_back(y[index]);
    }
    return {X_sample, y_sample};
}

// Train
void Random_Forest::fit(float_matrix &X_train, float_vector &y_train) {
#ifdef ENABLE_MPI
    int trees_per_process = n_trees / MPI_N_PROCESSES;
    int remaining_trees = n_trees % MPI_N_PROCESSES;
    int n_trees_for_main = trees_per_process + (remaining_trees > 0 ? 1 : 0);
        
    for(int i = 1; i < MPI_N_PROCESSES; ++i) {
        int n_trees_to_send = trees_per_process + (i <= remaining_trees ? 1 : 0);
        
        for(int j = 0; j < n_trees_to_send; j++) {
            auto [X_sample, y_sample] = bootstrap_sample(X_train, y_train);
            // cout << "Process: 0 - X_sample.size(): " << X_sample.size() << endl;
            
            vector<char> buf;
            serialize_matrix(buf, X_sample);
            serialize_vector(buf, y_sample);
            int buf_size = buf.size();
            
            MPI_Send(&buf_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(buf.data(), buf.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }

    for(int i = 0; i < n_trees_for_main; i++) {
        auto [X_sample, y_sample] = bootstrap_sample(X_train, y_train);
        trees[i]->fit(X_sample, y_sample);
    }
#else
    for (Cart* tree : trees) {
        auto [X_sample, y_sample] = bootstrap_sample(X_train, y_train);
        tree->fit(X_sample, y_sample);
    }
#endif
}

// Predict
float_vector Random_Forest::predict(float_matrix &X_test) {
    vector<float_vector> all_predictions;

#ifdef ENABLE_MPI
    int trees_per_process = n_trees / MPI_N_PROCESSES;
    int remaining_trees = n_trees % MPI_N_PROCESSES;
    int n_trees_for_main = trees_per_process + (remaining_trees > 0 ? 1 : 0);
    
    for(int i = 1; i < MPI_N_PROCESSES; i++) {
        vector<char> buf;
        serialize_matrix(buf, X_test);
        int buf_size = buf.size();
        MPI_Send(&buf_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(buf.data(), buf.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
    }

    for(int i = 0; i < n_trees_for_main; i++) {
        trees_preds.push_back(trees[i]->predict(X_test));
    }

    for(int i = 1; i < MPI_N_PROCESSES; i++) {
        int n_trees_to_send = trees_per_process + (i <= remaining_trees ? 1 : 0);
        
        for(int j = 0; j < n_trees_to_send; j++) {
            int buf_size;
            vector<char> buf;
            vector<float> pred;
            size_t idx = 0;
            
            MPI_Recv(&buf_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            buf.resize(buf_size);
            MPI_Recv(buf.data(), buf_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            deserialize_vector(buf, pred, idx);
            all_predictions.push_back(pred);
        }
    }

    for(int i = 0; i < n_trees_for_main; i++) {
        all_predictions.push_back(trees_preds[i]);
    }
#else
    for(Cart* tree : trees) {
        all_predictions.push_back(tree->predict(X_test));
    }
#endif

    float_vector predictions;
    for (size_t i = 0; i < X_test.size(); ++i) {
        unordered_map<float, int> class_counts;

        for (const auto &preds : all_predictions) {
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
