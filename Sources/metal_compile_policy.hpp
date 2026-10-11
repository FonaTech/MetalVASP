#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef METAL_VASP_DEFAULT_PROFILE
#define METAL_VASP_DEFAULT_PROFILE "auto"
#endif
#ifndef METAL_VASP_DEFAULT_LANGUAGE
#define METAL_VASP_DEFAULT_LANGUAGE "auto"
#endif
#ifndef METAL_VASP_FP32_FAST_MATH
#define METAL_VASP_FP32_FAST_MATH 0
#endif

namespace metal_policy {
// Deployment remains macOS 13+. An older SDK may compile this source using
// the highest baseline it declares; do not name unavailable enum constants.
inline MTLLanguageVersion baseline_msl() {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
    return MTLLanguageVersion2_4;
#else
    return MTLLanguageVersion2_3;
#endif
}
inline const char *profile() {
    const char *value = std::getenv("VASP_METAL_PROFILE");
    if (!value || !*value) value = METAL_VASP_DEFAULT_PROFILE;
    // An invalid runtime override selects the conservative profile.
    return std::strcmp(value,"auto")==0 || std::strcmp(value,"advanced")==0 ? value : "portable";
}
inline bool metal4(id<MTLDevice> device) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) return [device supportsFamily:MTLGPUFamilyMetal4];
#endif
    return false;
}
inline bool advanced(id<MTLDevice> device) {
    if (std::strcmp(profile(),"portable")==0) return false;
    const char *variant=std::getenv("VASP_METAL_DEVICE_VARIANT");
    if (variant && (std::strcmp(variant,"generic")==0 || std::strcmp(variant,"off")==0)) return false;
    // The implementation uses ordinary compute and shared tiles, not tensor/ANE.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
    return device.hasUnifiedMemory && [device supportsFamily:MTLGPUFamilyApple7] &&
           device.maxThreadgroupMemoryLength >= 8192 && device.maxThreadsPerThreadgroup.width >= 256;
#else
    return false;
#endif
}
inline void math_options(MTLCompileOptions *options, bool fp32_fast=false) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        options.mathMode = fp32_fast ? MTLMathModeFast : MTLMathModeSafe;
        options.mathFloatingPointFunctions = fp32_fast ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
        return;
    }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = fp32_fast ? YES : NO;
#pragma clang diagnostic pop
}
inline std::string language_name(MTLLanguageVersion value) {
    const auto v=static_cast<unsigned long>(value);
    return std::to_string(v>>16)+"."+std::to_string(v&65535);
}
inline std::vector<MTLLanguageVersion> languages(id<MTLDevice> device) {
    std::vector<MTLLanguageVersion> supported;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) if (metal4(device)) supported.push_back(MTLLanguageVersion4_0);
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) supported.push_back(MTLLanguageVersion3_2);
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
    if (@available(macOS 14.0, *)) supported.push_back(MTLLanguageVersion3_1);
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
    if (@available(macOS 13.0, *)) supported.push_back(MTLLanguageVersion3_0);
#endif
    supported.push_back(baseline_msl());
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
    // Explicit 2.3 also obeys the requested ceiling on a newer SDK. This is
    // the baseline known to the oldest SDK accepted for Apple Silicon builds.
    supported.push_back(MTLLanguageVersion2_3);
#endif
    const char *requested=std::getenv("VASP_METAL_LANGUAGE_VERSION");
    if (!requested || !*requested) requested=METAL_VASP_DEFAULT_LANGUAGE;
    if (std::strcmp(profile(),"portable")==0) {
        // A conservative profile still obeys an explicit lower ceiling. The
        // oldest accepted SDK declares 2.3, so no unavailable enum is named.
        if (std::strcmp(requested,"2.3")==0) return {MTLLanguageVersion2_3};
        return {baseline_msl()};
    }
    if (std::strcmp(requested,"auto")==0) return supported;
    // Explicit versions are attempted only when SDK/OS/device permit them.
    // Never fall forward to a newer language than the user requested.
    unsigned long ceiling=0;
    if(std::strcmp(requested,"2.3")==0) ceiling=(2u<<16)+3;
    else if(std::strcmp(requested,"2.4")==0) ceiling=(2u<<16)+4;
    else if(std::strcmp(requested,"3.0")==0) ceiling=3u<<16;
    else if(std::strcmp(requested,"3.1")==0) ceiling=(3u<<16)+1;
    else if(std::strcmp(requested,"3.2")==0) ceiling=(3u<<16)+2;
    else if(std::strcmp(requested,"4.0")==0) ceiling=4u<<16;
    for (size_t i=0;i<supported.size();++i)
        if (static_cast<unsigned long>(supported[i])<=ceiling) return {supported.begin()+i,supported.end()};
    return {baseline_msl()};
}
struct Compilation {
    id<MTLLibrary> library=nil;
    MTLLanguageVersion language=baseline_msl();
    unsigned failures=0;
    std::string error;
};
inline Compilation compile(id<MTLDevice> device, NSString *source,
                           NSDictionary<NSString *,NSObject *> *macros=nil,
                           bool fp32_fast=false) {
    Compilation result;
    for (auto language:languages(device)) {
        MTLCompileOptions *options=[MTLCompileOptions new];
        options.languageVersion=language;
        options.preprocessorMacros=macros;
        math_options(options,fp32_fast);
        NSError *error=nil;
        result.library=[device newLibraryWithSource:source options:options error:&error];
        result.language=language;
        if (result.library) return result;
        ++result.failures;
        result.error=error ? [[error localizedDescription] UTF8String] : "Metal library compilation failed";
    }
    return result;
}
}
