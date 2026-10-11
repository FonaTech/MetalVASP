#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace metal_nonlocal_policy {
inline bool multiply(size_t a,size_t b,size_t &out) {
    if(b && a>std::numeric_limits<size_t>::max()/b) return false;
    out=a*b;return true;
}
inline bool span(size_t rows,size_t columns,size_t ld,size_t width) {
    size_t value;
    return rows && columns && ld>=rows && multiply(ld,columns,value) && multiply(value,width,value);
}
// Only a dispatch window is32-bit. The logical shape and all buffer offsets
// stay64-bit; split work instead of rejecting a valid larger logical product.
inline size_t bands_per_dispatch(size_t sites,size_t requested) {
    if(!sites || sites>UINT32_MAX) return 0;
    return requested<UINT32_MAX/sites ? requested:UINT32_MAX/sites;
}
}
