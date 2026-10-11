#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include "metal_chunk_policy.hpp"

namespace metal_gemm {
using metal_chunk::multiply;
enum class Result { ok, invalid_shape, resource_limit };
struct Plan {
    size_t rows=0,cols=0,inner=0,atomic_host_bytes=0;
    std::array<size_t,3> bytes{};
    bool chunked=false;
};
inline bool host_shape(size_t m,size_t n,size_t k,bool trans_a,bool trans_b,
                       size_t lda,size_t ldb,size_t ldc) {
    size_t span=0;
    if(!m || !n || !k || lda<(trans_a ? k:m) || ldb<(trans_b ? n:k) || ldc<m)return false;
    return multiply(lda,trans_a ? m:k,span) && span<=SIZE_MAX/16 &&
           multiply(ldb,trans_b ? k:n,span) && span<=SIZE_MAX/16 &&
           multiply(ldc,n,span) && span<=SIZE_MAX/16;
}
inline Result plan(size_t m,size_t n,size_t k,bool ta,bool tb,size_t lda,size_t ldb,size_t ldc,
                   size_t max_buffer,size_t working,Plan &p) {
    p={};if(!host_shape(m,n,k,ta,tb,lda,ldb,ldc))return Result::invalid_shape;
    working=working ? working:max_buffer;
    size_t mk=0,kn=0,mn=0,total=0;
    const size_t dim_limit=size_t(UINT32_MAX)-15; // tiled dispatch edge lanes
    if(multiply(m,k,mk) && multiply(k,n,kn) && multiply(m,n,mn) &&
       mk<=SIZE_MAX/16 && kn<=SIZE_MAX/16 && mn<=SIZE_MAX/16) {
        std::array<size_t,3> bytes{mk*16,kn*16,mn*16};
        if(bytes[0]<=max_buffer && bytes[1]<=max_buffer && bytes[2]<=max_buffer &&
           bytes[0]<=SIZE_MAX-bytes[1] && (total=bytes[0]+bytes[1])<=SIZE_MAX-bytes[2] &&
           total+bytes[2]<=working && m<=dim_limit && n<=dim_limit && k<=dim_limit) {
            p.rows=m;p.cols=n;p.inner=k;p.bytes=bytes;return Result::ok;
        }
    }
    if(!multiply(m,n,mn) || !multiply(mn,16,p.atomic_host_bytes))return Result::invalid_shape;
    p.chunked=true;
    if(p.atomic_host_bytes>=working || max_buffer<16)return Result::resource_limit;
    const size_t available=working-p.atomic_host_bytes;
    // Maximize the output panel first; its partial sums remain on the GPU
    // through every K panel. Only A/B are refilled between submissions.
    p.rows=std::min(m,dim_limit);p.cols=std::min(n,dim_limit);
    for(;;) {
        size_t count=0,cbytes=0,sum=0;
        const bool fits=multiply(p.rows,p.cols,count) && multiply(count,16,cbytes) &&
                        cbytes<=max_buffer && cbytes<=available/3 &&
                        p.rows<=SIZE_MAX-p.cols && multiply(p.rows+p.cols,16,sum);
        if(fits) {
            p.inner=std::min({k,dim_limit,max_buffer/16/p.rows,max_buffer/16/p.cols,(available-cbytes)/sum});
            if(p.inner) {
                p.bytes={p.rows*p.inner*16,p.inner*p.cols*16,cbytes};return Result::ok;
            }
        }
        if(p.rows==1 && p.cols==1)return Result::resource_limit;
        if(p.rows>=p.cols && p.rows>1)p.rows=(p.rows+1)/2;
        else p.cols=(p.cols+1)/2;
    }
}
}
