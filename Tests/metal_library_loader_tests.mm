#include "metal_library_loader.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
 @autoreleasepool {
    using namespace metal_library;
    NSString *directory=[NSTemporaryDirectory() stringByAppendingPathComponent:[@"metal-offline-metadata-" stringByAppendingString:NSUUID.UUID.UUIDString]];
    assert([[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:nil]);
    NSString *path=[directory stringByAppendingPathComponent:@"backend.metallib"];
    NSString *manifest_path=[directory stringByAppendingPathComponent:@"build.json"];
    const char *pieces[]={"kernel void cpu_fixture() {}", "", "", "", "", "", "", "", ""};
    NSString *source=[NSString stringWithUTF8String:pieces[0]];
    Expectations expected;expected.language_ceiling="4.0";expected.os_version={{26,0,0}};
    auto run=[&](){return load(nil,path.UTF8String,source,pieces,expected);};
    assert(run().reason=="library-read-failed");
    assert(load(nil,"relative.metallib",source,pieces,expected).reason=="library-path-not-absolute-utf8");
    NSData *bytes=[@"not-a-Metal-library" dataUsingEncoding:NSUTF8StringEncoding];
    assert([bytes writeToFile:path atomically:YES]);assert(run().reason=="manifest-read-failed");
    assert([[@"{" dataUsingEncoding:NSUTF8StringEncoding] writeToFile:manifest_path atomically:YES]);
    assert(run().reason=="manifest-json-invalid");
    NSMutableArray *parts=[NSMutableArray array];
    for(size_t i=0;i<part_names.size();++i) [parts addObject:@{
        @"symbol":[NSString stringWithUTF8String:part_names[i]],
        @"shader_sha256":[NSString stringWithUTF8String:hash([[NSString stringWithUTF8String:pieces[i]] dataUsingEncoding:NSUTF8StringEncoding]).c_str()]
    }];
    NSMutableDictionary *good=[@{
        @"compilation_success":@YES,@"runtime_source_assembly_verified":@YES,
        @"status":@"offline-compilation-succeeded",@"shader_sha256":[NSString stringWithUTF8String:hash([source dataUsingEncoding:NSUTF8StringEncoding]).c_str()],
        @"metallib_sha256":[NSString stringWithUTF8String:hash(bytes).c_str()],@"language":@"2.4",@"deployment_target":@"13.0",
        @"compiler_flags":@[@"-fmetal-math-mode=safe",@"-fmetal-math-fp32-functions=precise",@"-ffp-contract=off"],
        @"macros":@{@"METAL_VASP_DEVICE_TUNED":@0,@"METAL_VASP_METAL4":@0},
        @"source_parts":parts,@"kernel_names":@[@"cpu_fixture"],
        @"commands":@[@{@"returncode":@0,@"argv":@[@"xcrun",@"metal",@"-v"]},
        @{@"returncode":@0,@"argv":@[@"xcrun",@"metal",@"-std=macos-metal2.4",@"-mmacosx-version-min=13.0",@"-fmetal-math-mode=safe",@"-fmetal-math-fp32-functions=precise",@"-ffp-contract=off",@"-DMETAL_VASP_DEVICE_TUNED=0",@"-DMETAL_VASP_METAL4=0",@"-c",@"source.metal",@"-o",@"output.air"]},
        @{@"returncode":@0,@"argv":@[@"xcrun",@"metallib",@"output.air",@"-o",@"backend.metallib"]}]
    } mutableCopy];
    auto write=[&](NSDictionary *m){assert([[NSJSONSerialization dataWithJSONObject:m options:0 error:nil] writeToFile:manifest_path atomically:YES]);};
    write(good);auto loaded=run();assert(loaded.verified && !loaded.library && loaded.reason.find("library-load-failed:")==0);
    NSMutableDictionary *bad=[good mutableCopy];bad[@"compiler_flags"]=@[@"-fmetal-math-mode=fast",@"-fmetal-math-fp32-functions=precise",@"-ffp-contract=off"];write(bad);
    assert(!run().verified && run().reason=="unsafe-compiler-flags");
    bad=[good mutableCopy];bad[@"shader_sha256"]=@"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";write(bad);assert(run().reason=="shader-sha256-mismatch");
    bad=[good mutableCopy];bad[@"metallib_sha256"]=@"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";write(bad);assert(run().reason=="metallib-sha256-mismatch");
    bad=[good mutableCopy];bad[@"macros"]=@{@"METAL_VASP_DEVICE_TUNED":@NO,@"METAL_VASP_METAL4":@NO};write(bad);assert(run().reason=="profile-macros-mismatch");
    bad=[good mutableCopy];bad[@"compilation_success"]=@"true";write(bad);assert(run().reason=="compilation-not-successful");
    bad=[good mutableCopy];[bad removeObjectForKey:@"deployment_target"];write(bad);assert(run().reason=="deployment-target-missing-or-invalid");
    bad=[good mutableCopy];bad[@"source_parts"]=@[];write(bad);assert(run().reason=="source-parts-incomplete");
    assert([[NSFileManager defaultManager] removeItemAtPath:directory error:nil]);
    std::puts("offline loader JSON/hash/invalid-file gates passed; CPU only, nil Metal device");
 }
}
