#include "metal_memory_allocator.hpp"
#import <Foundation/Foundation.h>
#include "metal4_dispatch.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#if !defined(METAL_VASP_DISABLE_METAL4) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000 && __has_include(<Metal/MTL4CommandQueue.h>)
#define METAL_VASP_COMPILED_METAL4 1
#else
#define METAL_VASP_COMPILED_METAL4 0
#endif

namespace {
void describe(char *out, size_t capacity, const char *message) {
    if (out && capacity) std::snprintf(out, capacity, "%s", message ? message : "unknown Metal 4 error");
}
}

struct metal4_dispatch {
#if METAL_VASP_COMPILED_METAL4
    id<MTLDevice> device=nil;
    // Type-erased ownership avoids referencing newer protocols in the public
    // context layout. Every API access below is inside the macOS 26 guard.
    id queue=nil, allocator=nil, command=nil, arguments=nil;
    id<MTLComputePipelineState> generic=nil, tiled=nil;
    id residency=nil;
    id<MTLBuffer> constants=nil;
    id<MTLBuffer> bound[4]{};
    std::mutex mutex;
#endif
};

#if METAL_VASP_COMPILED_METAL4
namespace {
id<MTLComputePipelineState> make_pipeline(id<MTL4Compiler> compiler,
                                          id<MTLLibrary> library, NSString *name,
                                          NSError **error) API_AVAILABLE(macos(26.0)) {
    // A legacy compute pipeline is not used as a Metal 4 pipeline. The new
    // compiler resolves the very same library function through its descriptor.
    MTL4LibraryFunctionDescriptor *function=[MTL4LibraryFunctionDescriptor new];
    function.library=library; function.name=name;
    MTL4ComputePipelineDescriptor *descriptor=[MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor=function;
    descriptor.maxTotalThreadsPerThreadgroup=256;
    return [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:error];
}
}
#endif

metal4_dispatch *metal4_dispatch_create(id<MTLDevice> device, id<MTLLibrary> library,
                                       char *error, size_t capacity) {
    describe(error,capacity,"");
#if METAL_VASP_COMPILED_METAL4
    if (@available(macOS 26.0, *)) {
        if (!device || !library || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            describe(error,capacity,"device does not support the Metal 4 command API"); return nullptr;
        }
        auto *d=new(std::nothrow) metal4_dispatch;
        if (!d) { describe(error,capacity,"Metal 4 context allocation failed"); return nullptr; }
        NSError *problem=nil;
        d->device=device; d->queue=[device newMTL4CommandQueue];
        d->allocator=[device newCommandAllocator]; d->command=[device newCommandBuffer];
        MTL4ArgumentTableDescriptor *arguments=[MTL4ArgumentTableDescriptor new];
        arguments.maxBufferBindCount=4; arguments.initializeBindings=YES;
        d->arguments=[device newArgumentTableWithDescriptor:arguments error:&problem];
        MTLResidencySetDescriptor *resources=[MTLResidencySetDescriptor new];
        resources.initialCapacity=4;
        d->residency=[device newResidencySetWithDescriptor:resources error:&problem];
        id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&problem];
        if (compiler) d->generic=make_pipeline(compiler,library,@"ds_zgemm",&problem);
        if (!d->queue || !d->allocator || !d->command || !d->arguments || !d->residency || !d->generic) {
            describe(error,capacity,problem ? problem.localizedDescription.UTF8String : "Metal 4 command object or generic pipeline creation failed");
            delete d; return nullptr;
        }
        // A library built for a generic-only device need not contain a tile.
        if ([library.functionNames containsObject:@"ds_zgemm_tiled"]) {
            problem=nil; d->tiled=make_pipeline(compiler,library,@"ds_zgemm_tiled",&problem);
        }
        return d;
    }
#endif
    (void)device; (void)library;
    describe(error,capacity,"Metal 4 requires a macOS 26 SDK build and a macOS 26 runtime");
    return nullptr;
}

void metal4_dispatch_destroy(metal4_dispatch *d) { delete d; }

bool metal4_dispatch_has_tiled(const metal4_dispatch *d) {
#if METAL_VASP_COMPILED_METAL4
    if (@available(macOS 26.0, *)) return d && d->tiled;
#endif
    (void)d;
    return false;
}

int metal4_dispatch_zgemm(metal4_dispatch *d, bool tiled,
                         id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                         const void *parameters, size_t bytes,
                         size_t m, size_t n, size_t threads,
                         double *gpu_seconds, double *wait_seconds,
                         char *error, size_t capacity) {
    if (gpu_seconds) *gpu_seconds=0.; if (wait_seconds) *wait_seconds=0.;
    describe(error,capacity,"");
#if METAL_VASP_COMPILED_METAL4
    if (@available(macOS 26.0, *)) {
        if (!d || !a || !b || !c || !parameters || !bytes || !m || !n || !threads ||
            m>std::numeric_limits<size_t>::max()/n ||
            m>std::numeric_limits<size_t>::max()-15 || n>std::numeric_limits<size_t>::max()-15 ||
            a.device!=d->device || b.device!=d->device || c.device!=d->device ||
            (tiled && !d->tiled)) {
            describe(error,capacity,"invalid Metal 4 GEMM resources, shape, or pipeline"); return 1;
        }
        std::lock_guard<std::mutex> lock(d->mutex);
        id<MTLComputePipelineState> pipeline=tiled ? d->tiled : d->generic;
        if (threads>pipeline.maxTotalThreadsPerThreadgroup) {
            describe(error,capacity,"threadgroup size exceeds Metal 4 pipeline capacity"); return 1;
        }
        if (!d->constants || d->constants.length<bytes)
            d->constants=metal_memory::allocate(d->device,bytes,MTLResourceStorageModeShared);
        if (!d->constants) { describe(error,capacity,"Metal 4 constant buffer allocation failed"); return 2; }
        std::memcpy(d->constants.contents,parameters,bytes);
        // Metal 4 does not infer residency from argument-table GPU addresses.
        // The set owns these allocations through completion of this command.
        // Production staging buffers are usually reused. Preserve the exact
        // allocation set and bindings until an allocation changes, rather
        // than rebuilding residency metadata on every GEMM call.
        if(d->bound[0]!=a || d->bound[1]!=b || d->bound[2]!=c || d->bound[3]!=d->constants) {
            [d->residency removeAllAllocations];
            [d->residency addAllocation:a]; [d->residency addAllocation:b];
            [d->residency addAllocation:c]; [d->residency addAllocation:d->constants];
            [d->residency commit];
            d->bound[0]=a; d->bound[1]=b; d->bound[2]=c; d->bound[3]=d->constants;
            for(NSUInteger i=0;i<4;++i) [d->arguments setAddress:d->bound[i].gpuAddress atIndex:i];
        }
        [d->command beginCommandBufferWithAllocator:d->allocator];
        [d->command useResidencySet:d->residency];
        id<MTL4ComputeCommandEncoder> encoder=[(id<MTL4CommandBuffer>)d->command computeCommandEncoder];
        if (!encoder) {
            [d->command endCommandBuffer]; [d->allocator reset];
            describe(error,capacity,"Metal 4 compute encoder creation failed"); return 3;
        }
        [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:d->arguments];
        if (tiled) [encoder dispatchThreadgroups:MTLSizeMake((n+15)/16,(m+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        else [encoder dispatchThreads:MTLSizeMake(n,m,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        [encoder endEncoding]; [d->command endCommandBuffer];
        dispatch_semaphore_t finished=dispatch_semaphore_create(0);
        __block NSError *problem=nil; __block double elapsed=0.;
        MTL4CommitOptions *options=[MTL4CommitOptions new];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            problem=feedback.error;
            if(feedback.GPUEndTime>feedback.GPUStartTime) elapsed=feedback.GPUEndTime-feedback.GPUStartTime;
            dispatch_semaphore_signal(finished);
        }];
        id<MTL4CommandBuffer> commands[]={d->command};
        const auto start=std::chrono::steady_clock::now();
        [d->queue commit:commands count:1 options:options];
        // Completion feedback is the host visibility boundary. The allocator,
        // argument table, and constants are not reused until GPU work finishes.
        dispatch_semaphore_wait(finished,DISPATCH_TIME_FOREVER);
        if (wait_seconds) *wait_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if (gpu_seconds) *gpu_seconds=elapsed;
        [d->allocator reset];
        if (problem) { describe(error,capacity,problem.localizedDescription.UTF8String); return 3; }
        const char *fault=std::getenv("VASP_METAL_TEST_METAL4_FAILURE");
        if(fault && std::strcmp(fault,"after-completion")==0) {
            // Explicit failure-atomicity probe: C now contains a computed
            // result, exactly the state a failed command might leave behind.
            describe(error,capacity,"injected Metal 4 failure after GPU completion"); return 3;
        }
        return 0;
    }
#endif
    (void)d; (void)tiled; (void)a; (void)b; (void)c; (void)parameters;
    (void)bytes; (void)m; (void)n; (void)threads;
    describe(error,capacity,"Metal 4 command API was not compiled or is unavailable");
    return 4;
}
