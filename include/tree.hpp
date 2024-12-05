#ifndef _TREE_HPP_
#define _TREE_HPP_

#include "utils.hpp"

void cuda_best_threshold_sender(const float_matrix &X, const float_vector &y, int num_samples, int num_features, int* best_feature, float* best_threshold);

#endif
