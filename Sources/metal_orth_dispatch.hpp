#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
namespace metal_orth {
inline bool cooperative(size_t npl,size_t npro,bool available) {
    if (!available) return false;
    const char *mode=std::getenv("VASP_METAL_ORTH_KERNEL");
    if (mode && !std::strcmp(mode,"serial")) return false;
    if (mode && !std::strcmp(mode,"cooperative")) return true;
    // Performance route only. Small inputs still use the original kernel;
    // this threshold does not reject any matrix size.
    return npl>=512 || npro>=512;
}
}
