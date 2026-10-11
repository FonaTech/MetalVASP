#include "metal_full_orth_policy.hpp"
#include <cassert>
#include <cstdio>
using namespace metal_full_orth;
int main() {
    Plan p;
    const size_t gib=size_t(1024)*1024*1024;
    assert(plan(1024,8,320,2,4,1030,12,gib,4*gib,p)==Result::ok);
    assert(p.bytes[0]==5242880 && p.bytes[6]==1638400 && p.bytes[4]==160);
    assert(p.total_bytes==13888692);
    assert(plan(1024,8,320,2,4,1030,12,4096,4*gib,p)==Result::resource_limit);
    assert(plan(1024,8,320,2,4,1030,12,gib,1000000,p)==Result::resource_limit);
    assert(plan(1024,8,320,2,4,1030,12,gib,0,p)==Result::ok);
    assert(plan(0,0,320,0,0,0,0,gib,gib,p)==Result::invalid_shape);
    assert(plan(1024,8,320,2,4,1023,12,gib,gib,p)==Result::invalid_shape);
    assert(plan(1024,8,320,2,4,1030,7,gib,gib,p)==Result::invalid_shape);
    assert(plan(1024,8,320,0,4,1030,12,gib,gib,p)==Result::invalid_shape);
    assert(plan(1,0,65536,0,0,1,0,SIZE_MAX,SIZE_MAX,p)==Result::ok);
    assert(plan(1,0,SIZE_MAX,0,0,1,0,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    assert(plan(SIZE_MAX,1,2,1,1,SIZE_MAX,1,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    assert(plan(1,1,1,SIZE_MAX,2,1,1,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    assert(plan(1,1,2,1,1,SIZE_MAX,1,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    // With no projectors, the PAW metadata is unused and has no allocation.
    assert(plan(640,0,512,10,20,644,0,gib,gib,p)==Result::ok);
    assert(p.bytes[3]==16 && p.bytes[4]==20);
    std::puts("full orth policy: 320/512 bands, resource budgets, 64-bit offsets and overflow gates passed");
}
