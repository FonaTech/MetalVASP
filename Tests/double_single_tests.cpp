#include "double_single.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

int main() {
    const std::array<double, 8> values = {
        0.0, 1.0, -1.0, 1e-20, -1e-9, 0.125, 12.3456789012345, 1e20
    };
    for (double value : values) {
        const double observed = DoubleSingle::split(value).value();
        assert(std::abs(observed - value) <= std::max(std::abs(value) * 2e-15, 1e-45));
    }

    const std::array<std::pair<double, double>, 4> cases = {{
        {1.23456789012345, 0.000000123456789012345},
        {1e12, -1e12 + 0.25},
        {-0.75, 0.125},
        {1e-12, 2e-12},
    }};
    for (const auto &[lhs, rhs] : cases) {
        const auto a = DoubleSingle::split(lhs);
        const auto b = DoubleSingle::split(rhs);
        const double sum = ds_add(a, b).value();
        const double product = ds_mul(a, b).value();
        assert(std::abs(sum - (lhs + rhs)) <= std::max(std::abs(lhs + rhs) * 3e-14, 1e-40));
        assert(std::abs(product - (lhs * rhs)) <= std::max(std::abs(lhs * rhs) * 3e-14, 1e-40));
    }

    std::mt19937_64 random(20261001);
    std::uniform_real_distribution<double> mantissa(-1.0, 1.0);
    std::uniform_int_distribution<int> exponent(-20, 20);
    for (int index = 0; index < 10000; ++index) {
        const double lhs = std::ldexp(mantissa(random), exponent(random));
        const double rhs = std::ldexp(mantissa(random), exponent(random));
        const auto a = DoubleSingle::split(lhs);
        const auto b = DoubleSingle::split(rhs);
        const double sum = ds_add(a, b).value();
        const double product = ds_mul(a, b).value();
        // DS has about 48 significant bits: cancellation amplifies relative
        // error in the result, so use an input-scaled forward-error bound.
        assert(std::abs(sum - (lhs + rhs)) <= std::max((std::abs(lhs) + std::abs(rhs)) * 5e-14, 1e-40));
        assert(std::abs(product - (lhs * rhs)) <= std::max(std::abs(lhs * rhs) * 5e-14, 1e-40));
    }

    const auto outside_float_exponent_range = DoubleSingle::split(1e100);
    assert(!std::isfinite(outside_float_exponent_range.value()));
    const auto nan_value = DoubleSingle::split(std::numeric_limits<double>::quiet_NaN());
    assert(std::isnan(nan_value.value()));
}
