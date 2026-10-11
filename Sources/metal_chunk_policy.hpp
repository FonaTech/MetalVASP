#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace metal_chunk {
enum class Result { ok, invalid_shape, resource_limit };
struct FFTPlan { size_t total=0,chunk_batches=0,buffer_bytes=0,atomic_host_bytes=0,root_bytes=0; bool chunked=false; };
inline bool multiply(size_t a,size_t b,size_t &v) {if(a && b>SIZE_MAX/a)return false;v=a*b;return true;}
inline Result fft(size_t nx,size_t ny,size_t nz,size_t batches,size_t stride,
                  size_t max_buffer,size_t working,FFTPlan &p) {
    p={};size_t volume=0,span=0,one=0,whole=0;
    if(!nx || !ny || !nz || !batches || !multiply(nx,ny,volume) || !multiply(volume,nz,volume) ||
       nx>UINT32_MAX || ny>UINT32_MAX || nz>UINT32_MAX || stride<volume || !multiply(stride,batches,span) || span>SIZE_MAX/16 ||
       !multiply(volume,16,one) || !multiply(one,batches,whole))return Result::invalid_shape;
    p.total=volume;working=working ? working:max_buffer;
    // Twiddle allocations are shared by axes of equal length. This budget
    // covers the current sign; older cached roots are evicted under pressure.
    const size_t dimensions[3]={nx,ny,nz};
    for(size_t i=0;i<3;++i)if(dimensions[i]>1) {
        bool seen=false;for(size_t j=0;j<i;++j)seen|=dimensions[j]==dimensions[i];
        if(!seen) {
            size_t bytes=0;if(!multiply(dimensions[i],16,bytes))return Result::invalid_shape;
            if(bytes>max_buffer || p.root_bytes>SIZE_MAX-bytes)return Result::resource_limit;
            p.root_bytes+=bytes;
        }
    }
    if(p.root_bytes>=working)return Result::resource_limit;
    working-=p.root_bytes;
    const size_t dispatch_limit=size_t(UINT32_MAX)*65536;
    if(whole<=max_buffer && whole<=working/2 && batches<=dispatch_limit/volume) {
        p.chunk_batches=batches;p.buffer_bytes=whole;return Result::ok;
    }
    // Output is staged once on the host so a later chunk failure cannot corrupt
    // the caller's input before the legacy CPU fallback. Each source element
    // is imported once, each result exported once; no per-axis host boundaries.
    p.chunked=true;p.atomic_host_bytes=whole;
    if(whole>=working || one>max_buffer || volume>dispatch_limit)return Result::resource_limit;
    const size_t count=std::min({batches,max_buffer/one,(working-whole)/2/one,dispatch_limit/volume});
    if(!count)return Result::resource_limit;
    p.chunk_batches=count;p.buffer_bytes=count*one;return Result::ok;
}
}
