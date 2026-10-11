#pragma once
#include <cstddef>
#include <cstdint>

namespace metal_memory {
// `allocated` is the device's aggregate allocation report, not a single
// operator's cache. CPU arrays and memory used by other processes are outside
// this policy; callers must not report this as a system-memory guarantee.
inline bool can_allocate(uint64_t allocated, size_t requested, uint64_t budget,
                         size_t max_buffer) {
    return requested && requested <= max_buffer && allocated <= budget &&
           uint64_t(requested) <= budget - allocated;
}
}
