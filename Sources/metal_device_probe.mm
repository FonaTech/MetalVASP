#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "metal_compile_policy.hpp"
#include "metal_backend.h"

static void describe_device(FILE *out, id<MTLDevice> device, NSUInteger index) {
    std::fprintf(out, "device[%lu].name=%s\n", (unsigned long)index, [[device name] UTF8String]);
    std::fprintf(out, "device[%lu].registry_id=%llu\n", (unsigned long)index, (unsigned long long)[device registryID]);
    std::fprintf(out, "device[%lu].low_power=%s\n", (unsigned long)index, [device isLowPower] ? "true" : "false");
    std::fprintf(out, "device[%lu].headless=%s\n", (unsigned long)index, [device isHeadless] ? "true" : "false");
    std::fprintf(out, "device[%lu].unified_memory=%s\n", (unsigned long)index, [device hasUnifiedMemory] ? "true" : "false");
    const MTLSize threads = [device maxThreadsPerThreadgroup];
    std::fprintf(out, "device[%lu].max_threads_dimensions=%lu,%lu,%lu\n", (unsigned long)index,
                 (unsigned long)threads.width, (unsigned long)threads.height, (unsigned long)threads.depth);
    std::fprintf(out, "device[%lu].max_threadgroup_memory_bytes=%lu\n", (unsigned long)index, (unsigned long)[device maxThreadgroupMemoryLength]);
    std::fprintf(out, "device[%lu].max_buffer_bytes=%lu\n", (unsigned long)index, (unsigned long)[device maxBufferLength]);
    std::fprintf(out, "device[%lu].recommended_working_set_bytes=%llu\n", (unsigned long)index, (unsigned long long)[device recommendedMaxWorkingSetSize]);
    std::fprintf(out, "device[%lu].current_allocated_bytes=%lu\n", (unsigned long)index, (unsigned long)[device currentAllocatedSize]);
    // Family values 1001..1010 are stable Metal ABI values. Unknown newer
    // families are queried without referring to SDK enum names, and return NO
    // on older devices. The name is diagnostic only, never a dispatch key.
    std::fprintf(out, "device[%lu].apple_gpu_families=", (unsigned long)index);
    bool first = true;
    for (NSInteger family = 1; family <= 10; ++family) {
        if ([device supportsFamily:(MTLGPUFamily)(1000 + family)]) {
            std::fprintf(out, "%s%ld", first ? "" : ",", (long)family);
            first = false;
        }
    }
    std::fprintf(out, "\n");
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        std::fprintf(out, "device[%lu].metal4_family=%s\n", (unsigned long)index,
                     [device supportsFamily:MTLGPUFamilyMetal4] ? "true" : "false");
    } else
#endif
    {
        std::fprintf(out, "device[%lu].metal4_family=unqueried_sdk_or_os\n", (unsigned long)index);
    }
}

static bool describe_pipeline(FILE *out, id<MTLDevice> device) {
    NSError *error = nil;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
                        "kernel void capability_probe(device float *x [[buffer(0)]], "
                        "uint i [[thread_position_in_grid]]) { x[i] = x[i] + 1.0f; }\n";
    auto compiled=metal_policy::compile(device,source);
    id<MTLLibrary> library=compiled.library;
    std::fprintf(out,"compile.profile=%s\ncompile.language=%s\ncompile.math=safe-precise\ncompile.fallbacks=%u\n",
                 metal_policy::profile(),metal_policy::language_name(compiled.language).c_str(),compiled.failures);
    id<MTLFunction> function = [library newFunctionWithName:@"capability_probe"];
    id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    std::fprintf(out, "probe_pipeline.compiles=%s\n", pipeline ? "true" : "false");
    if (!pipeline) {
        std::fprintf(out, "probe_pipeline.error=%s\n", error ? [[error localizedDescription] UTF8String] : "function unavailable");
        return false;
    }
    // These are properties of this simple probe, not every VASP kernel.
    std::fprintf(out, "probe_pipeline.thread_execution_width=%lu\n", (unsigned long)pipeline.threadExecutionWidth);
    std::fprintf(out, "probe_pipeline.max_total_threads=%lu\n", (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
    std::fprintf(out, "probe_pipeline.static_threadgroup_memory_bytes=%lu\n", (unsigned long)pipeline.staticThreadgroupMemoryLength);
    std::fprintf(out, "probe_pipeline.executed=false\n");
    return true;
}

int main() {
    @autoreleasepool {
        const char *out_path = std::getenv("METAL_PROBE_OUTPUT");
        FILE *out = out_path ? std::fopen(out_path, "w") : stdout;
        if (!out) out = stdout;
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        std::fprintf(out, "default_device=%s\n", default_device ? [[default_device name] UTF8String] : "<none>");
        std::fprintf(out, "enumerated_devices=%lu\n", (unsigned long)devices.count);
        for (NSUInteger i = 0; i < devices.count; ++i) {
            describe_device(out, devices[i], i);
        }
        id<MTLDevice> selected = default_device;
        if (!selected) {
            for (id<MTLDevice> candidate in devices) {
                if (![candidate isLowPower]) { selected = candidate; break; }
            }
            if (!selected) selected = devices.firstObject;
        }
        std::fprintf(out, "selected_device=%s\n", selected ? [[selected name] UTF8String] : "<none>");
        // A default device is not necessarily included in the enumeration.
        if (selected && ![devices containsObject:selected]) describe_device(out, selected, devices.count);
        std::fprintf(out, "m5_specialized_kernel_implemented=false\n");
        std::fprintf(out, "routing_calibrated=false\n");
        bool pipeline_ok = selected ? describe_pipeline(out, selected) : false;
        if (pipeline_ok) {
            metal_backend *backend=nullptr;
            metal_backend_create(&backend,1);
            std::fprintf(out,"backend.gpu_enabled=%d\nbackend.variant=%s\nbackend.language=%s\nbackend.tiled_gemm=%d\nbackend.compile_fallbacks=%u\nbackend.tiled_fallbacks=%llu\n",
                metal_backend_gpu_enabled(backend),metal_backend_variant(backend),metal_backend_language_version(backend),
                metal_backend_tiled_gemm_enabled(backend),metal_backend_compile_fallbacks(backend),
                (unsigned long long)metal_backend_tiled_gemm_fallbacks(backend));
            if (!metal_backend_gpu_enabled(backend)) std::fprintf(out,"backend.error=%s\n",metal_backend_last_error(backend));
            pipeline_ok=metal_backend_gpu_enabled(backend);
            std::fprintf(out,"backend.tiled_gemm_autoselected=%d\nbackend.gemm_calibrated=%d\n",metal_backend_tiled_gemm_autoselected(backend),metal_backend_gemm_calibrated(backend));
            metal_backend_destroy(backend);
        }
        if (out != stdout) std::fclose(out);
        if (!selected) {
            std::fprintf(stderr, "No Metal device is visible to this process. The system profiler can still list a GPU when the process is outside a GUI login session or has GPU access restricted.\n");
            return 2;
        }
        if (!pipeline_ok) return 3;
    }
    return 0;
}
