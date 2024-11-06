#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

using namespace std;

#define float_matrix vector<vector<float>>
#define float_vector vector<float>

void trainTestSplit(const float_matrix& X, const float_vector& y,
                    float_matrix& X_train, float_vector& y_train,
                    float_matrix& X_test, float_vector& y_test,
                    int random_state, float test_size);

float calculateAccuracy(const float_vector& y_true, const float_vector& y_pred);

#endif
