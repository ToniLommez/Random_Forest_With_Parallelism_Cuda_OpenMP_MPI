#include <iostream>
#include <chrono>
#include <random>
#include "include/utils.hpp"
#include "include/RandomForest.hpp"
#include "include/reader.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

using namespace std;

#ifdef ENABLE_MPI
void init_tree(int rank) {
    vector<char> buf;
    int buf_size;

    // ----------------------------------------------------------
    // Init
    // ----------------------------------------------------------
    
    int n_trees;
    MPI_Recv(&n_trees, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // cout << "Process: " << rank << " - n_trees: " << n_trees << endl;
    
    int max_depth, min_samples_split;
    bool isClassification;
    vector<Cart*> trees;
    trees.resize(n_trees);

    for(int i = 0; i < n_trees; i++) {
        MPI_Recv(&max_depth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&min_samples_split, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&isClassification, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        trees[i] = new Cart(max_depth, min_samples_split, isClassification);
        // cout << "Process: " << rank << " - Received tree data of index: " << i << endl;
    }
    
    // ----------------------------------------------------------
    // Train
    // ----------------------------------------------------------
    
    vector<vector<float>> y_train;
    vector<vector<vector<float>>> X_train;
    y_train.resize(n_trees);
    X_train.resize(n_trees);

    for(int i = 0; i < n_trees; i++) {
        MPI_Recv(&buf_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        buf.resize(buf_size);
        MPI_Recv(buf.data(), buf_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // cout << "Process: " << rank << " - Received train data idx:" << i << endl;

        size_t idx = 0;
        deserialize_matrix(buf, X_train[i], idx);
        // cout << "Process: " << rank << " - X_sample.size(): " << X_train.size() << endl;
        deserialize_vector(buf, y_train[i], idx);
        buf.clear();
    }

    for(int i = 0; i < n_trees; i++) {
        trees[i]->fit(X_train[i], y_train[i]);
    }
    
    // ----------------------------------------------------------
    // Predict
    // ----------------------------------------------------------
    
    // cout << "gaming" << endl;
    MPI_Recv(&buf_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    buf.resize(buf_size);
    MPI_Recv(buf.data(), buf_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<vector<float>> X_test;
    size_t idx = 0;
    deserialize_matrix(buf, X_test, idx);
    
    vector<vector<float>> preds;
    preds.resize(n_trees);
    
    for(int i = 0; i < n_trees; i++) {
        preds[i] = trees[i]->predict(X_test);
    }

    for(int i = 0; i < n_trees; i++) {
        serialize_vector(buf, preds[i]);
        buf_size = buf.size();
        
        MPI_Send(&buf_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buf.data(), buf_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        buf.clear();
    }
}
#endif

int main(int argc, char* argv[]) {
    
#ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank == 0) {
#endif
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
    Random_Forest rf(10, 15, 2, true);
    rf.fit(X_train, y_train);
    
    // Fazer predições no conjunto de teste
    float_vector y_pred = rf.predict(X_test);

    // Calcular a Acurácia
    float accuracy = calculateAccuracy(y_test, y_pred);
    cout << "Acuracia: " << accuracy * 100 << "%" << endl;
    
#ifdef ENABLE_MPI
    } // end if(rank == 0);
    else {
        init_tree(rank);
    }
    
    MPI_Finalize();
#endif

    return 0;
}
