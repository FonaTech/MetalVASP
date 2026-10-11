#include "metal_chunk_policy.hpp"
#include <cassert>
#include <cstdio>
int main() {
    using namespace metal_chunk;FFTPlan p;
    assert(fft(5,4,3,5,64,1920,65536,p)==Result::ok && p.chunked && p.chunk_batches==2 && p.atomic_host_bytes==4800);
    assert(fft(5,4,3,5,64,8192,65536,p)==Result::ok && !p.chunked && p.chunk_batches==5);
    // A total batch beyond32-bit is legal: two-dimensional dispatch is bounded.
    const size_t batches=size_t(UINT32_MAX)/64+1;
    assert(fft(4,4,4,batches,64,64*1024*1024,SIZE_MAX,p)==Result::ok && p.chunked);
    assert(fft(65536,65536,1,1,size_t(65536)*65536,SIZE_MAX,SIZE_MAX,p)==Result::ok && !p.chunked);
    assert(fft(5,4,3,5,64,128,65536,p)==Result::resource_limit);
    assert(fft(5,4,3,5,64,1920,4800,p)==Result::resource_limit);
    assert(fft(5,4,3,5,59,1920,65536,p)==Result::invalid_shape);
    assert(fft(SIZE_MAX,2,2,1,SIZE_MAX,SIZE_MAX,SIZE_MAX,p)==Result::invalid_shape);
    std::puts("FFT chunk policy: wide total batches, bounded windows, atomic staging and overflow passed");
}
