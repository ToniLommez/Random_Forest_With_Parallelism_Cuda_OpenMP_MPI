#include <iostream>
#include <chrono>
#include <random>
#include "include/utils.hpp"
#include "include/randomForest.hpp"
#include "include/reader.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    // Verificar se o caminho para o CSV foi fornecido como argumento
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " -csv <caminho_para_csv>" << endl;
        return 1;
    }

    string csv_path;

    // Processar argumentos
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-csv" && i + 1 < argc) {
            csv_path = argv[i + 1];
        }
    }

    // Verificar se o caminho do CSV foi atribuído
    if (csv_path.empty()) {
        cerr << "Erro: O caminho para o CSV nao foi especificado." << endl;
        return 1;
    }

    // Ler o arquivo CSV
    Reader reader(csv_path);
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
    randomForest rf(10);
    rf.train(X_train, y_train);

    // Fazer predições no conjunto de teste
    float_vector y_pred = rf.predict(X_test);

    // Calcular a Acurácia
    float accuracy = calculateAccuracy(y_test, y_pred);
    cout << "Acuracia: " << accuracy * 100 << "%" << endl;

    return 0;
}
