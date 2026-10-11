#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include "double_single.hpp"
#include "metal4_dispatch.h"
#include "metal_stats_path.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

using cd=std::complex<double>;
struct Parameters { uint32_t m,n,k; vector_float4 alpha,beta; };
vector_float4 pack(cd value) {
 const auto r=DoubleSingle::split(value.real()), i=DoubleSingle::split(value.imag());
 return {r.hi,r.lo,i.hi,i.lo};
}
int main(int argc,char **argv) {
 if(argc<2 || argc>3 || (argc==3 && std::strcmp(argv[2],"--metal4")!=0)) {
  std::fprintf(stderr,"usage: offline-library-probe /absolute/backend.metallib [--metal4]\n"); return 2;
 }
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if(!device) return 77;
  NSError *error=nil; NSURL *url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
  id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
  if(!library) { std::fprintf(stderr,"offline load: %s\n",error.localizedDescription.UTF8String); return 1; }
  const bool use4=argc==3;
  metal4_dispatch *modern=nullptr; char problem[2048]{};
  id<MTLCommandQueue> queue=nil; id<MTLComputePipelineState> pipeline=nil;
  if(use4) {
   modern=metal4_dispatch_create(device,library,problem,sizeof(problem));
   if(!modern) { std::fprintf(stderr,"offline Metal 4 pipeline: %s\n",problem); return 1; }
  } else {
   pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"ds_zgemm"] error:&error];
   queue=[device newCommandQueue];
   if(!pipeline || !queue) return 1;
  }
  const size_t m=33,n=47,k=29; const cd alpha(.7,-.3),beta(-.2,.4);
  std::vector<cd> a(m*k),b(k*n),c(m*n),reference(m*n);
  for(size_t i=0;i<a.size();++i) a[i]={std::sin(.13*i),std::cos(.17*i)};
  for(size_t i=0;i<b.size();++i) b[i]={std::cos(.11*i),std::sin(.19*i)};
  for(size_t i=0;i<c.size();++i) c[i]={std::sin(.07*i),std::cos(.23*i)};
  for(size_t row=0;row<m;++row) for(size_t col=0;col<n;++col) {
   cd sum{}; for(size_t inner=0;inner<k;++inner) sum+=a[row*k+inner]*b[inner*n+col];
   reference[row*n+col]=alpha*sum+beta*c[row*n+col];
  }
  id<MTLBuffer> ab=[device newBufferWithLength:a.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb=[device newBufferWithLength:b.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
  id<MTLBuffer> cb=[device newBufferWithLength:c.size()*sizeof(vector_float4) options:MTLResourceStorageModeShared];
  if(!ab||!bb||!cb) {metal4_dispatch_destroy(modern);return 1;}
  for(size_t i=0;i<a.size();++i) static_cast<vector_float4 *>(ab.contents)[i]=pack(a[i]);
  for(size_t i=0;i<b.size();++i) static_cast<vector_float4 *>(bb.contents)[i]=pack(b[i]);
  for(size_t i=0;i<c.size();++i) static_cast<vector_float4 *>(cb.contents)[i]=pack(c[i]);
  Parameters p{uint32_t(m),uint32_t(n),uint32_t(k),pack(alpha),pack(beta)};
  double gpu=0.,wait=0.;
  const auto start=std::chrono::steady_clock::now();
  if(use4) {
   if(metal4_dispatch_zgemm(modern,false,ab,bb,cb,&p,sizeof(p),m,n,256,&gpu,&wait,problem,sizeof(problem))) {
    std::fprintf(stderr,"offline Metal 4 dispatch: %s\n",problem);metal4_dispatch_destroy(modern);return 1;
   }
  } else {
   id<MTLCommandBuffer> command=[queue commandBuffer]; id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
   [encoder setComputePipelineState:pipeline]; [encoder setBuffer:ab offset:0 atIndex:0];
   [encoder setBuffer:bb offset:0 atIndex:1]; [encoder setBuffer:cb offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
   [encoder dispatchThreads:MTLSizeMake(m*n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
   [encoder endEncoding];[command commit];[command waitUntilCompleted];
   if(command.status!=MTLCommandBufferStatusCompleted) {metal4_dispatch_destroy(modern);return 1;}
   gpu=command.GPUEndTime-command.GPUStartTime;
  }
  double host=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  double maximum=0.; bool finite=true;
  for(size_t i=0;i<c.size();++i) {
   auto x=static_cast<vector_float4 *>(cb.contents)[i]; cd actual{double(x.x)+x.y,double(x.z)+x.w};
   finite=finite&&std::isfinite(actual.real())&&std::isfinite(actual.imag()); maximum=std::max(maximum,std::abs(actual-reference[i]));
  }
  bool passed=finite && maximum<1e-8 && gpu>0.;
  auto names=[library.functionNames sortedArrayUsingSelector:@selector(compare:)];
  std::printf("{\"device\":%s,\"library\":%s,\"command_api\":%d,\"shape\":[33,47,29],\"host_s\":%.9g,\"gpu_s\":%.9g,\"max_abs_error\":%.9g,\"passed\":%s,\"kernel_names\":[",metal_stats::json_string(device.name.UTF8String).c_str(),metal_stats::json_string(argv[1]).c_str(),use4?4:3,host,gpu,maximum,passed?"true":"false");
  for(NSUInteger i=0;i<names.count;++i) std::printf("%s%s",i?",":"",metal_stats::json_string([names[i] UTF8String]).c_str());
  std::puts("]}"); metal4_dispatch_destroy(modern); return passed?0:1;
 }
}
