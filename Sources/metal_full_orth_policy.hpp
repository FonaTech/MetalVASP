#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace metal_full_orth {
enum class Result { ok, invalid_shape, resource_limit };
struct Plan { std::array<size_t,12> bytes{}; size_t total_bytes=0; };
inline bool product(size_t a,size_t b,size_t &out) {
    if(a && b>std::numeric_limits<size_t>::max()/a) return false;
    out=a*b;return true;
}
// Matrix addresses use ulong; dimensions retain the existing uint ABI.
// Both buffer lengths and the aggregate retained workspace must fit device
// resources; allocation and numerical failure remain separate fallback gates.
inline Result plan(size_t npl,size_t npro,size_t bands,size_t nions,size_t lm,
                   size_t ldw,size_t ldp,size_t max_buffer,size_t max_working,Plan &out) {
    out={};
    if(!npl || !bands || ldw<npl || ldp<npro ||
       (npro && (!nions || !lm || npro>size_t(INT32_MAX)))) return Result::invalid_shape;
    size_t wave=0,proj=0,gram=0,metric=0,span=0,host_span=0;
    if(!product(npl,bands,wave) || !product(npro,bands,proj) ||
       !product(bands,bands,gram) || npl>SIZE_MAX-npro ||
       !product(npl+npro,bands,span) ||
       !product(ldw,bands,host_span) || host_span>SIZE_MAX/16 ||
       !product(ldp,bands,host_span) || host_span>SIZE_MAX/16 ||
       (npro && (!product(nions,lm,metric) || !product(metric,lm,metric))))
        return Result::invalid_shape;
    for(auto value:{npl,npro,nions,lm,bands})
        if(value>UINT32_MAX) return Result::invalid_shape;
    const std::array<size_t,12> count{wave,proj,proj,metric,npro,bands,gram,gram,1,wave,proj,1};
    // The site ABI is five uint32_t fields, including the unused type field.
    const std::array<size_t,12> width{16,16,16,16,20,8,16,16,4,16,16,16};
    for(size_t i=0;i<count.size();++i) {
        size_t bytes=0;
        if(!product(count[i] ? count[i]:1,width[i],bytes)) return Result::invalid_shape;
        if(bytes>max_buffer || out.total_bytes>SIZE_MAX-bytes) return Result::resource_limit;
        out.bytes[i]=bytes;out.total_bytes+=bytes;
    }
    // A zero recommendation is unknown: use the conservative per-buffer cap
    // as the aggregate budget, rather than guessing the available RAM.
    if(out.total_bytes>(max_working ? max_working:max_buffer)) return Result::resource_limit;
    return Result::ok;
}
}
