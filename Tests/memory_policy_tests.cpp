#include "metal_memory_policy.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
int main() {
    using metal_memory::can_allocate;
    assert(can_allocate(60,40,100,100));
    assert(!can_allocate(60,41,100,100));
    assert(!can_allocate(101,1,100,100));
    assert(!can_allocate(0,101,200,100));
    assert(!can_allocate(0,0,100,100));
    assert(can_allocate(UINT64_MAX-16,16,UINT64_MAX,SIZE_MAX));
    assert(!can_allocate(UINT64_MAX-16,17,UINT64_MAX,SIZE_MAX));
    std::puts("Aggregate budget boundary and overflow checks passed");
}
