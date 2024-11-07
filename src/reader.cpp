#include "../include/reader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>


using namespace std;
// Constructor
Reader::Reader(const string &filepath) : filepath(filepath) {}

// readCSV reads the CSV, splitting it into X and Y
// the last line is used as Y
bool Reader::readCSV() {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening the file: " << filepath << endl;
        return false;
    }

    string line;

    // Process header line
    if (getline(file, line)) {
        columnNames = splitHeader(line);
    }

    // Process data lines
    while (getline(file, line)) {
        vector<float> row = splitLine(line);

        if (row.empty()) {
            continue;
        }

        // Split row into X and Y
        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }

    file.close();
    return true;
}

// Get X matrix
const float_matrix &Reader::getX() const {
    return X;
}

// Get Y vector
const float_vector &Reader::getY() const {
    return y;
}

// Get column names
const vector<string>& Reader::getColumnNames() const {
    return columnNames;
}

// Split a line of CSV into a vector of floats
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
        } catch (const out_of_range &e) { // Catch out-of-range exceptions
            cerr << "Float value out of range: " << cell << endl;
            values.push_back(0.0);
        }
    }

    return values;
}

// Split header line into column names
vector<string> Reader::splitHeader(const string& line) {
    vector<string> headers;
    stringstream ss(line);
    string cell;

    while (getline(ss, cell, ',')) {
        headers.push_back(cell);
    }

    return headers;
}
