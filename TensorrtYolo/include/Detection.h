#pragma once
#include <vector>

struct Detection {
    int class_id;
    float score;
    int left;
    int top;
    int width;
    int height;

    bool operator<(const Detection& other) const {
        return score > other.score;
    }
};