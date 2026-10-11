#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace metal_eigh {
enum class Result { ok, invalid_shape, resource_limit };
struct Plan {
    size_t matrix_bytes=0, rotation_bytes=0;
    size_t device_bytes=0, host_bytes=0, peak_workspace_bytes=0;
};
inline bool multiply(size_t a,size_t b,size_t &value) {
    if(a && b>SIZE_MAX/a) return false;
    value=a*b;return true;
}
inline bool add(size_t a,size_t b,size_t &value) {
    if(b>SIZE_MAX-a) return false;
    value=a+b;return true;
}
// 64-bit matrix offsets and size_t host spans are checked independently.
// Workspace estimates include ABI temporaries and independent FP64 checks.
inline Result plan(size_t n,size_t lda,size_t ldb,size_t ldv,bool generalized,
                   size_t max_buffer,size_t working_budget,Plan &p) {
    p={};size_t elements=0,span=0;
    if(!n || lda<n || ldv<n || (generalized && ldb<n) ||
       !multiply(n,n,elements) || n>UINT32_MAX ||
       !multiply(lda,n,span) || span>SIZE_MAX/16 ||
       !multiply(ldv,n,span) || span>SIZE_MAX/16 ||
       (generalized && (!multiply(ldb,n,span) || span>SIZE_MAX/16)))
        return Result::invalid_shape;
    if(!multiply(elements,16,p.matrix_bytes) ||
       !multiply((n+(n%2))/2,64,p.rotation_bytes)) return Result::invalid_shape;
    if(p.matrix_bytes>max_buffer || p.rotation_bytes>max_buffer || max_buffer<8)
        return Result::resource_limit;
    // Include both cyclic-Jacobi and optional LOBPCG peaks. Small n cannot
    // bound18*n LOBPCG columns with a constant matrix-count estimate.
    size_t cyclic=0,lob=0,extra=0,base=0;
    if(!multiply(p.matrix_bytes,generalized ? 11:5,cyclic) ||
       !add(p.rotation_bytes,12,extra) || !add(cyclic,extra,cyclic) ||
       !multiply(p.matrix_bytes,generalized ? 9:3,lob) ||
       !multiply(n,18*16+8,extra) || !add(extra,44,extra) || !add(lob,extra,lob))
        return Result::resource_limit;
    p.device_bytes=cyclic>lob ? cyclic:lob;
    if(!multiply(p.matrix_bytes,generalized ? 5:4,p.host_bytes) ||
       !multiply(n,16,extra) || !add(p.host_bytes,extra,p.host_bytes) ||
       !add(p.device_bytes,p.host_bytes,p.peak_workspace_bytes)) return Result::resource_limit;
    if(p.peak_workspace_bytes>(working_budget ? working_budget:max_buffer)) return Result::resource_limit;
    return Result::ok;
}
inline long long saturated_cubic(int n) {
    if(n<=0) return 0;
    const auto x=static_cast<unsigned long long>(n);
    const auto maximum=static_cast<unsigned long long>(std::numeric_limits<long long>::max());
    if(x>maximum/x || x*x>maximum/x) return std::numeric_limits<long long>::max();
    return static_cast<long long>(x*x*x);
}
}
