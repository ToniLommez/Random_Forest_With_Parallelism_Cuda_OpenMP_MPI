#include "../include/node.hpp"
#include "../include/utils.hpp"

using namespace std;

Node::Node(const int feature, const bool is_leaf, const float threshold, const float prediction)
    : feature(feature), is_leaf(is_leaf), threshold(threshold), prediction(prediction){

    left = nullptr;
    right = nullptr;
}

Node::Node() {
    left = nullptr;
    right = nullptr;
}

Node::~Node() {
    delete left;
    delete right;
}
