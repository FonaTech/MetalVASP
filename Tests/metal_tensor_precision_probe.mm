#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_compile_policy.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// Deliberately independent of the production DS ABI. This experiment asks
// whether an ordinary half-input/float-output MPP matmul can replace FP64/DS.
// It is not a VASP tensor backend, and it does not identify hardware counters
// for the M5 neural accelerator or the separate Apple Neural Engine.
static const char *source=R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
struct Shape { int m,n,k; };
// MPP's value-type dispatch strips address spaces but not const. The input
// tensor value type must be half rather than const half; this kernel never
// writes either input buffer.
kernel void half_tensor_product(device half *a [[buffer(0)]],
                                device half *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant Shape &p [[buffer(3)]],
                                uint2 group [[threadgroup_position_in_grid]]) {
 auto left=tensor(a,dextents<int,2>{p.k,p.m},array<int,2>{1,p.k});
 auto right=tensor(b,dextents<int,2>{p.n,p.k},array<int,2>{1,p.n});
 auto output=tensor(c,dextents<int,2>{p.n,p.m},array<int,2>{1,p.n});
 auto tile_a=left.slice(0,int(group.y)*16);
 auto tile_b=right.slice(int(group.x)*16,0);
 auto tile_c=output.slice(int(group.x)*16,int(group.y)*16);
 constexpr auto descriptor=tensor_ops::matmul2d_descriptor(16,16);
 tensor_ops::matmul2d<descriptor,execution_simdgroup> operation;
 operation.run(tile_a,tile_b,tile_c);
}
)METAL";

int main() {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
 @autoreleasepool {
  if(@available(macOS 26.0,*)) {
   id<MTLDevice> device=MTLCreateSystemDefaultDevice();
   if(!device || ![device supportsFamily:MTLGPUFamilyMetal4]) return 77;
   NSError *error=nil;
   id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
   MTLCompileOptions *options=[MTLCompileOptions new]; options.languageVersion=MTLLanguageVersion4_0;
   metal_policy::math_options(options,false);
   MTL4LibraryDescriptor *library_descriptor=[MTL4LibraryDescriptor new];
   library_descriptor.source=[NSString stringWithUTF8String:source]; library_descriptor.options=options;
   id<MTLLibrary> library=[compiler newLibraryWithDescriptor:library_descriptor error:&error];
   if(!library) { std::fprintf(stderr,"MPP library: %s\n",error.localizedDescription.UTF8String); return 1; }
   MTL4LibraryFunctionDescriptor *function=[MTL4LibraryFunctionDescriptor new];
   function.library=library; function.name=@"half_tensor_product";
   MTL4ComputePipelineDescriptor *pipeline_descriptor=[MTL4ComputePipelineDescriptor new];
   pipeline_descriptor.computeFunctionDescriptor=function;
   pipeline_descriptor.requiredThreadsPerThreadgroup=MTLSizeMake(32,1,1);
   pipeline_descriptor.maxTotalThreadsPerThreadgroup=32;
   id<MTLComputePipelineState> pipeline=[compiler newComputePipelineStateWithDescriptor:pipeline_descriptor compilerTaskOptions:nil error:&error];
   if(!pipeline) { std::fprintf(stderr,"MPP pipeline: %s\n",error.localizedDescription.UTF8String); return 1; }
   id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];
   id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
   id<MTL4CommandBuffer> command=[device newCommandBuffer];
   const int m=96,n=80,k=64; struct Shape { int m,n,k; } shape{m,n,k};
   std::vector<double> a(m*k),b(k*n),reference(m*n),rounded_reference(m*n);
   std::mt19937_64 rng(20261009); std::uniform_real_distribution<double> draw(-1.,1.);
   for(auto &v:a) v=draw(rng); for(auto &v:b) v=draw(rng);
   id<MTLBuffer> ab=[device newBufferWithLength:a.size()*sizeof(_Float16) options:MTLResourceStorageModeShared];
   id<MTLBuffer> bb=[device newBufferWithLength:b.size()*sizeof(_Float16) options:MTLResourceStorageModeShared];
   id<MTLBuffer> cb=[device newBufferWithLength:reference.size()*sizeof(float) options:MTLResourceStorageModeShared];
   id<MTLBuffer> pb=[device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared];
   if(!queue||!allocator||!command||!ab||!bb||!cb||!pb) return 1;
   for(size_t i=0;i<a.size();++i) static_cast<_Float16 *>(ab.contents)[i]=_Float16(a[i]);
   for(size_t i=0;i<b.size();++i) static_cast<_Float16 *>(bb.contents)[i]=_Float16(b[i]);
   for(int row=0;row<m;++row) for(int col=0;col<n;++col) for(int inner=0;inner<k;++inner) {
    reference[row*n+col]+=a[row*k+inner]*b[inner*n+col];
    rounded_reference[row*n+col]+=double(static_cast<_Float16 *>(ab.contents)[row*k+inner])*double(static_cast<_Float16 *>(bb.contents)[inner*n+col]);
   }
   MTL4ArgumentTableDescriptor *arguments_descriptor=[MTL4ArgumentTableDescriptor new]; arguments_descriptor.maxBufferBindCount=4;
   id<MTL4ArgumentTable> arguments=[device newArgumentTableWithDescriptor:arguments_descriptor error:&error];
   MTLResidencySetDescriptor *residency_descriptor=[MTLResidencySetDescriptor new]; residency_descriptor.initialCapacity=4;
   id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:residency_descriptor error:&error];
   if(!arguments||!residency) return 1;
   [arguments setAddress:ab.gpuAddress atIndex:0]; [arguments setAddress:bb.gpuAddress atIndex:1];
   [arguments setAddress:cb.gpuAddress atIndex:2]; [arguments setAddress:pb.gpuAddress atIndex:3];
   [residency addAllocation:ab]; [residency addAllocation:bb]; [residency addAllocation:cb]; [residency addAllocation:pb]; [residency commit];
   std::vector<double> host,gpu;
   for(int repeat=0;repeat<11;++repeat) {
    const auto start=std::chrono::steady_clock::now();
    [command beginCommandBufferWithAllocator:allocator]; [command useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:arguments];
    [encoder dispatchThreadgroups:MTLSizeMake((n+15)/16,(m+15)/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [encoder endEncoding]; [command endCommandBuffer];
    dispatch_semaphore_t finished=dispatch_semaphore_create(0); __block NSError *problem=nil; __block double seconds=0.;
    MTL4CommitOptions *commit=[MTL4CommitOptions new];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
     problem=feedback.error; seconds=feedback.GPUEndTime-feedback.GPUStartTime; dispatch_semaphore_signal(finished);
    }];
    id<MTL4CommandBuffer> commands[]={command}; [queue commit:commands count:1 options:commit];
    dispatch_semaphore_wait(finished,DISPATCH_TIME_FOREVER); [allocator reset];
    if(problem) { std::fprintf(stderr,"MPP GPU: %s\n",problem.localizedDescription.UTF8String); return 1; }
    if(repeat>=2) { gpu.push_back(seconds); host.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()); }
   }
   std::sort(host.begin(),host.end()); std::sort(gpu.begin(),gpu.end());
   double full_error=0.,rounded_error=0.,scale=1.; bool finite=true;
   for(size_t i=0;i<reference.size();++i) {
    const double actual=static_cast<float *>(cb.contents)[i]; finite=finite&&std::isfinite(actual);
    full_error=std::max(full_error,std::abs(actual-reference[i]));
    rounded_error=std::max(rounded_error,std::abs(actual-rounded_reference[i]));
    scale=std::max(scale,std::abs(rounded_reference[i]));
   }
   const bool implementation_gate=finite && rounded_error<=1e-5*scale && gpu[4]>0.;
   const bool ds_replacement_gate=finite && full_error<=1e-8;
   std::printf("{\"device\":\"%s\",\"kernel\":\"MPP matmul2d via MTL4ComputeCommandEncoder\",\"shape\":[96,80,64],\"inputs\":\"float16\",\"output\":\"float32\",\"warmup_repeats\":2,\"timed_repeats\":9,\"host_median_s\":%.9g,\"gpu_median_s\":%.9g,\"max_abs_error_vs_original_fp64\":%.9g,\"max_abs_error_vs_rounded_input_fp64\":%.9g,\"implementation_gate\":%s,\"ds_replacement_gate\":%s,\"production_enabled\":false,\"ane_execution_claimed\":false}\n",device.name.UTF8String,host[4],gpu[4],full_error,rounded_error,implementation_gate?"true":"false",ds_replacement_gate?"true":"false");
   return implementation_gate ? 0 : 1;
  }
 }
#endif
 return 77;
}
