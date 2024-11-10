#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>

using namespace std;

class Node {
public:
    Node(const int feature, const bool is_leaf, const float threshold, const float prediction);
    Node();
    ~Node();

    int feature;
    bool is_leaf;
    float threshold;
    float prediction;

    Node* left;
    Node* right;
};

#endif
