#pragma once
// Private layout shared only by owner implementations. No client owns an
// Objective-C resource or can mutate a device generation through this header.
#include "metal_wave_owner.h"
#include <memory>
#include <mutex>
struct metal_wave_owner {
 const metal_backend *backend=nullptr;metal_wave_owner_shape shape{};
 id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> library=nil;
 id<MTLComputePipelineState> precondition=nil,project=nil,metric=nil,norm=nil,normalize=nil;
 id<MTLBuffer> wave=nil,precon=nil,projection=nil,metric_projection=nil,q=nil,radial=nil,cq=nil,phase=nil,site=nil,scale=nil,status=nil;
 uint64_t accepted_generation=0,exported_generation=0;bool valid=false,static_valid=false,use_overlap=false;
 uint64_t control_status_host_bytes=0;
 metal_wave_owner_stats stats{};mutable std::mutex mutex;
 std::shared_ptr<void> davidson;
};
// Callers hold owner.mutex. Deferred preparation only encodes and consumes a
// generation; successful output remains invalid until its command completes.
int metal_wave_owner_prepare_locked(metal_wave_owner *,uint64_t,uint64_t,const metal_wave_prepare_input *,id<MTLCommandBuffer> __strong *deferred);
int metal_wave_owner_check_norm_locked(metal_wave_owner *);
