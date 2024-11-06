#include <iostream>
#include <chrono>
#include <random>
#include "include/utils.hpp"
#include "include/cart.hpp"
#include "include/reader.hpp"

using namespace std;

int main() {
    // Ler o arquivo CSV
    Reader reader("datasets/random.csv");
    if (!reader.readCSV()) {
        cerr << "Erro ao ler o arquivo CSV." << endl;
        return 1;
    }

    // Obter X e y dos dados lidos
    float_matrix X = reader.getX();
    float_vector y = reader.getY();

    // Dividir os dados em treino e teste
    float_matrix X_train, X_test;
    float_vector y_train, y_test;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine rng(seed);
    uniform_int_distribution<int> distribution(1, 100);
    trainTestSplit(X, y, X_train, y_train, X_test, y_test, distribution(rng), 0.3f);

    // Criar e Treinar a Árvore
    Cart cart;
    cart.fit(X_train, y_train);

    // Fazer predições no conjunto de teste
    float_vector y_pred = cart.predict(X_test);

    // Calcular a Acurácia
    float accuracy = calculateAccuracy(y_test, y_pred);
    cout << "Acuracia: " << accuracy * 100 << "%" << endl;

    return 0;
}
