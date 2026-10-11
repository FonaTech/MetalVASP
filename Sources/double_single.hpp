#pragma once

#include <cmath>

struct DoubleSingle {
    float hi;
    float lo;

    static DoubleSingle split(double value) {
        const float high = static_cast<float>(value);
        return {high, static_cast<float>(value - static_cast<double>(high))};
    }

    double value() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
};

inline DoubleSingle ds_add(DoubleSingle a, DoubleSingle b) {
    const float sum = a.hi + b.hi;
    const float virtual_b = sum - a.hi;
    const float error = ((a.hi - (sum - virtual_b)) + (b.hi - virtual_b)) + a.lo + b.lo;
    const float high = sum + error;
    return {high, error - (high - sum)};
}

inline DoubleSingle ds_mul(DoubleSingle a, DoubleSingle b) {
    const float product = a.hi * b.hi;
    const float error = std::fma(a.hi, b.hi, -product)
        + a.hi * b.lo
        + a.lo * b.hi
        + a.lo * b.lo;
    const float high = product + error;
    return {high, error - (high - product)};
}
