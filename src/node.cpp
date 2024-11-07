#include "../include/node.hpp"
#include "../include/utils.hpp"

using namespace std;
// Constructor with parameters
Node::Node(const int feature, const bool is_leaf, const float threshold, const float prediction)
    : feature(feature), is_leaf(is_leaf), threshold(threshold), prediction(prediction),
      left(nullptr), right(nullptr) {}

// Default constructor
Node::Node() : feature(0), is_leaf(false), threshold(0.0f), prediction(0.0f), left(nullptr), right(nullptr) {}

// Destructor
Node::~Node() {
    delete left;
    delete right;
}