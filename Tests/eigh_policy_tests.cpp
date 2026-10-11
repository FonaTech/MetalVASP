#include "metal_eigh_policy.hpp"
#include <cassert>
#include <cstdio>
using namespace metal_eigh;
int main() {
    Plan p;const size_t gib=size_t(1024)*1024*1024;
    assert(plan(129,134,0,132,false,gib,4*gib,p)==Result::ok);
    assert(p.matrix_bytes==129*129*16 && p.rotation_bytes==65*64);
    assert(plan(320,325,327,324,true,gib,4*gib,p)==Result::ok);
    assert(p.device_bytes==11*p.matrix_bytes+p.rotation_bytes+12);
    assert(plan(512,512,512,512,true,gib,4*gib,p)==Result::ok);
    assert(plan(320,320,320,320,true,4096,gib,p)==Result::resource_limit);
    assert(plan(320,320,320,320,true,gib,1024,p)==Result::resource_limit);
    assert(plan(0,0,0,0,false,gib,gib,p)==Result::invalid_shape);
    assert(plan(129,128,129,129,true,gib,gib,p)==Result::invalid_shape);
    assert(plan(129,129,128,129,true,gib,gib,p)==Result::invalid_shape);
    assert(plan(129,SIZE_MAX,129,129,true,gib,gib,p)==Result::invalid_shape);
    assert(plan(65536,65536,0,65536,false,SIZE_MAX,SIZE_MAX,p)==Result::ok);
    assert(plan(SIZE_MAX,SIZE_MAX,0,SIZE_MAX,false,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    assert(plan(2,2,0,2,false,gib,gib,p)==Result::ok);
    assert(p.device_bytes>=3*2*2*16+2*(18*16+8)+44);
    assert(saturated_cubic(320)==32768000);
    assert(saturated_cubic(INT32_MAX)==INT64_MAX && saturated_cubic(-1)==0);
    std::puts("EIGH policy: resource, host/kernel offset and saturated cost gates passed");
}
