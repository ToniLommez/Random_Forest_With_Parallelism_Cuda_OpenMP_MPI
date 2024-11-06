#ifndef READER_HPP
#define READER_HPP

#include "../include/utils.hpp"
#include <string>
#include <vector>

using namespace std;

class Reader {
  public:
    Reader(const string &filepath);

    bool readCSV();

    const float_matrix &getX() const;
    const float_vector &getY() const;
    const vector<string> &getColumnNames() const;

  private:
    string filepath;
    float_matrix X;
    float_vector y;
    vector<string> columnNames;

    vector<float> splitLine(const string &line);
    std::vector<std::string> splitHeader(const std::string& line);
};

#endif
