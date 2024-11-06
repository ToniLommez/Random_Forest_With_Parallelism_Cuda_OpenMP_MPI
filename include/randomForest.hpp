#include "cart.hpp"




class randomForest
{
private:
    
    struct tree
    {
        Cart cart;//arvore
    };
    
    int num_trees;
    vector<tree> forest;

    float evaluate(float_vector& predictions);

public:
    randomForest(int n);
    ~randomForest();

    void train(float_matrix &X_train, float_vector &y_train);
    float_vector predict(float_matrix& X_test);
};
