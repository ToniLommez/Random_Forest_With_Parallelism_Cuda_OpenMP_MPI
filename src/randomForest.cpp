#include "randomForest.hpp"
#include <algorithm>
#include <random>
#include <map>

/*

1. pegar parte treino
2. n = numero de arvores
3. para cada n{
    3.1. subconjunto = sortear x valores do treino e colocar para treino em uma árvore
                       e features
    3.2. para cada subconjunto criar uma árvore(PARALELIZAR)
    
    }

PREDIÇÃO
1. pega o dado e passa para todas as árvores simultaneamente


cada árvore, guardar as features utilizadas

*/

randomForest::randomForest(int n)
{

    num_trees = n;
    forest.resize(n);

}

randomForest::~randomForest()
{
}

void randomForest::train(float_matrix &X_train, float_vector &y_train){

    int n_features = X_train[0].size();
    int n_samples = X_train.size();

    for (int i = 0; i < num_trees; i++)
    {
        //sortear x valores do treino
        vector<int> random_indexes;
        for (int j = 0; j < n_samples; j++)
        {
            random_indexes.push_back(j);
        }
        shuffle(random_indexes.begin(), random_indexes.end(), default_random_engine(random_device{}()));

        std::random_device rd;                          // Usado para gerar uma semente
        std::mt19937 generator(rd());                   // Gerador de Mersenne Twister
        std::uniform_int_distribution<int> distribution(1, 100); // Intervalo de 1 a 100

        // Gerar um número aleatório
        int samples_used = distribution(generator);

        //pegar parte do treino
        float_matrix X_train_sub;
        float_vector y_train_sub;
        for (int j = 0; j < samples_used; j++)
        {
            X_train_sub.push_back(X_train[random_indexes[j]]);
            y_train_sub.push_back(y_train[random_indexes[j]]);
        }

        //treinar a árvore
        forest[i].cart.fit(X_train_sub, y_train_sub);
    }

}

float_vector randomForest::predict(float_matrix& X_test){

    vector<float_vector> predictions_global;
    float_vector predictions;
    for (size_t i = 0; i < forest.size(); i++)
    {
        predictions_global.push_back(forest[i].cart.predict(X_test));
    }

    for (size_t i = 0; i < X_test.size(); i++)
    {
        float_vector prediction_local;
        for (size_t j = 0; j < forest.size(); j++)
        {
            prediction_local.push_back(predictions_global[j][i]);
        }

        predictions.push_back(evaluate(prediction_local));
        
    }
    
    return predictions;
    
    
}

float randomForest::evaluate(float_vector& predictions){

    map<float, int> predictions_map;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        predictions_map[predictions[i]]++;
    }
    
    int max_count = 0;
    float max_class = 0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (int(predictions_map[predictions[i]]) > max_count)
        {
            max_count = predictions_map[predictions[i]];
            max_class = predictions[i];
        }
    }
    return max_class;
}