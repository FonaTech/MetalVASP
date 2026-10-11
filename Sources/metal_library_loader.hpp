#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <limits>
#include "metal_library_policy.hpp"

namespace metal_library {
inline std::string hash(NSData *data) {
    if(!data || data.length>std::numeric_limits<CC_LONG>::max()) return {};
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes,static_cast<CC_LONG>(data.length),bytes);
    static const char hex[]="0123456789abcdef";std::string result;
    for(unsigned char c:bytes) {result+=hex[c>>4];result+=hex[c&15];}
    return result;
}
inline std::string string(id value) {
    const char *bytes=[value isKindOfClass:NSString.class] ? [value UTF8String] : nullptr;
    return bytes ? std::string(bytes) : std::string();
}
inline bool boolean(id value) {
    return value && CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID() && [value boolValue];
}
inline int binary_integer(id value) {
    if(![value isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID()) return -1;
    const double number=[value doubleValue];return number==0.?0:number==1.?1:-1;
}
inline std::vector<std::string> strings(id value) {
    std::vector<std::string> result;
    if(![value isKindOfClass:NSArray.class]) return result;
    for(id entry in value) {auto text=string(entry);if(text.empty()) return {};result.push_back(text);}
    return result;
}
struct Loaded {
    id<MTLLibrary> library=nil;
    bool verified=false;
    std::string reason="not-requested",language;
};
inline Loaded load(id<MTLDevice> device,const char *path,NSString *source,
                   const char *const *pieces,const Expectations &profile) {
    Loaded result;
    if(!path || !*path) return result;
    NSString *filename=[NSString stringWithUTF8String:path];
    if(!filename || !filename.isAbsolutePath) {result.reason="library-path-not-absolute-utf8";return result;}
    NSError *error=nil;
    NSData *library_data=[NSData dataWithContentsOfFile:filename options:0 error:&error];
    if(!library_data) {result.reason="library-read-failed";return result;}
    NSString *manifest_path=[[filename stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"build.json"];
    NSData *manifest_data=[NSData dataWithContentsOfFile:manifest_path options:0 error:&error];
    if(!manifest_data) {result.reason="manifest-read-failed";return result;}
    id json=[NSJSONSerialization JSONObjectWithData:manifest_data options:0 error:&error];
    if(![json isKindOfClass:NSDictionary.class]) {result.reason="manifest-json-invalid";return result;}
    NSDictionary *manifest=json;
    Metadata m;
    m.compilation_success=boolean(manifest[@"compilation_success"]);
    m.assembly_verified=boolean(manifest[@"runtime_source_assembly_verified"]);
    m.status=string(manifest[@"status"]);m.shader_hash=string(manifest[@"shader_sha256"]);
    m.library_hash=string(manifest[@"metallib_sha256"]);m.language=string(manifest[@"language"]);
    m.deployment_target=string(manifest[@"deployment_target"]);
    m.flags=strings(manifest[@"compiler_flags"]);m.kernels=strings(manifest[@"kernel_names"]);
    id commands=manifest[@"commands"];
    if([commands isKindOfClass:NSArray.class] && [commands count]==3) {
        m.commands_success=true;
        for(id command in commands) {
            if(![command isKindOfClass:NSDictionary.class]) {m.commands_success=false;break;}
            id code=command[@"returncode"];
            if(![code isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)code)==CFBooleanGetTypeID() || [code doubleValue]!=0.) m.commands_success=false;
        }
        if(m.commands_success) {
            m.compiler_arguments=strings(commands[1][@"argv"]);
            const auto version_args=strings(commands[0][@"argv"]),link_args=strings(commands[2][@"argv"]);
            if(version_args!=std::vector<std::string>({"xcrun","metal","-v"}) || link_args.size()!=5 || link_args[0]!="xcrun" || link_args[1]!="metallib" || link_args[3]!="-o" || link_args[4].empty() || m.compiler_arguments.size()!=13 || link_args[2]!=m.compiler_arguments[12]) m.commands_success=false;
        }
    }
    id macros=manifest[@"macros"];
    if([macros isKindOfClass:NSDictionary.class] && [macros count]==2) {
        m.device_tuned=binary_integer(macros[@"METAL_VASP_DEVICE_TUNED"]);
        m.metal4=binary_integer(macros[@"METAL_VASP_METAL4"]);
    }
    id parts=manifest[@"source_parts"];
    if([parts isKindOfClass:NSArray.class]) for(id part in parts) {
        if(![part isKindOfClass:NSDictionary.class]) break;
        m.symbols.push_back(string(part[@"symbol"]));m.part_hashes.push_back(string(part[@"shader_sha256"]));
    }
    Expectations expected=profile;
    expected.shader_hash=hash([source dataUsingEncoding:NSUTF8StringEncoding]);
    expected.library_hash=hash(library_data);
    for(size_t i=0;i<part_names.size();++i) expected.part_hashes.push_back(hash([[NSString stringWithUTF8String:pieces[i]] dataUsingEncoding:NSUTF8StringEncoding]));
    NSRegularExpression *pattern=[NSRegularExpression regularExpressionWithPattern:@"\\bkernel\\s+void\\s+(\\w+)\\s*\\(" options:0 error:nil];
    for(NSTextCheckingResult *match in [pattern matchesInString:source options:0 range:NSMakeRange(0,source.length)]) expected.kernels.push_back(string([source substringWithRange:[match rangeAtIndex:1]]));
    result.reason=validate(m,expected);
    if(!result.reason.empty()) return result;
    result.verified=true;result.language=m.language;
    // Load the exact byte snapshot whose hash was checked, avoiding a second
    // pathname read that could select different content after validation.
    dispatch_data_t data=dispatch_data_create(library_data.bytes,library_data.length,dispatch_get_global_queue(QOS_CLASS_DEFAULT,0),^{(void)library_data;});
    result.library=[device newLibraryWithData:data error:&error];
    if(!result.library) {result.reason="library-load-failed:"+string(error.localizedDescription);return result;}
    auto loaded_names=strings(result.library.functionNames);
    std::sort(loaded_names.begin(),loaded_names.end());std::sort(expected.kernels.begin(),expected.kernels.end());
    if(loaded_names!=expected.kernels) {result.library=nil;result.reason="loaded-kernel-list-mismatch";return result;}
    // Check every kernel before accepting the library. A driver-incompatible
    // pipeline is rejected here, while runtime source fallback is still safe.
    for(NSString *name in result.library.functionNames) {
        auto function=[result.library newFunctionWithName:name];
        if(![device newComputePipelineStateWithFunction:function error:&error]) {
            result.library=nil;result.reason="library-pipeline-failed:"+string(name)+":"+string(error.localizedDescription);return result;
        }
    }
    result.reason="none";return result;
}
}
