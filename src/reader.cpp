#include "../include/reader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>


using namespace std;

Reader::Reader(const string &filepath) : filepath(filepath) {}

// readCSV read the csv spliting it in X and Y
// the last line is used as Y
bool Reader::readCSV() {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening the file: " << filepath << std::endl;
        return false;
    }

    std::string line;

    if (std::getline(file, line)) {
        columnNames = splitHeader(line);
    }

    while (std::getline(file, line)) {
        std::vector<float> row = splitLine(line);

        if (row.empty()) {
            continue;
        }

        y.push_back(row.back());
        row.pop_back();

        X.push_back(row);
    }

    file.close();
    return true;
}


const float_matrix &Reader::getX() const {
    return X;
}

const float_vector &Reader::getY() const {
    return y;
}

const std::vector<std::string>& Reader::getColumnNames() const {
    return columnNames;
}

vector<float> Reader::splitLine(const string &line) {
    vector<float> values;
    stringstream ss(line);
    string cell;

    while (getline(ss, cell, ',')) {
        try {
            values.push_back(stof(cell));
        } catch (const invalid_argument &e) {
            cerr << "Error converting to float: " << cell << endl;
            values.push_back(0.0);
        }
    }

    return values;
}

std::vector<std::string> Reader::splitHeader(const std::string& line) {
    std::vector<std::string> headers;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        headers.push_back(cell);
    }

    return headers;
}
