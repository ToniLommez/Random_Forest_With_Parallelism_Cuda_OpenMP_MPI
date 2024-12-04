#ifndef _TREE_HPP_
#define _TREE_HPP_

#include "RandomForest.hpp"

void cuda_init_trees(Cart** h_trees, int n_trees, int max_depth, int min_samples_split, bool isClassification);

#endif
