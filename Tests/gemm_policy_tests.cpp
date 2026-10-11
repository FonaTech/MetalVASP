#include "metal_gemm_policy.hpp"
#include <cassert>
#include <cstdio>
int main() {
    using namespace metal_gemm;Plan p;
    assert(plan(17,19,23,false,false,20,24,21,65536,65536,p)==Result::ok && !p.chunked);
    assert(plan(17,19,23,false,false,20,24,21,1024,65536,p)==Result::ok && p.chunked);
    for(auto bytes:p.bytes)assert(bytes<=1024);
    assert(p.inner<23 && p.atomic_host_bytes==17*19*16);
    // Planning a >32-bit flattened matrix performs no allocation.
    assert(plan(65536,65536,1,false,false,65536,1,65536,SIZE_MAX,SIZE_MAX,p)==Result::ok && !p.chunked);
    assert(plan(size_t(UINT32_MAX)+1,1,1,false,false,size_t(UINT32_MAX)+1,1,size_t(UINT32_MAX)+1,1024,SIZE_MAX,p)==Result::ok && p.chunked);
    assert(plan(17,19,23,true,true,22,19,17,1024,65536,p)==Result::invalid_shape);
    assert(plan(17,19,23,false,false,17,23,SIZE_MAX,1024,65536,p)==Result::invalid_shape);
    assert(plan(17,19,23,false,false,17,23,17,1024,17*19*16,p)==Result::resource_limit);
    std::puts("GEMM: wide host spans, bounded panels, retained K accumulation and resource gates passed");
}
