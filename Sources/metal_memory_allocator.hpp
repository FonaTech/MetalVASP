#pragma once
#import <Metal/Metal.h>
#include "metal_backend.h"
#include "metal_memory_policy.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace metal_memory {
inline std::vector<const std::mutex *> &held_operator_mutexes() {
    static thread_local std::vector<const std::mutex *> value;
    return value;
}
class OperatorGuard {
    std::lock_guard<std::mutex> guard;
public:
    explicit OperatorGuard(std::mutex &mutex) : guard(mutex) {
        held_operator_mutexes().push_back(&mutex);
    }
    ~OperatorGuard() { held_operator_mutexes().pop_back(); }
    OperatorGuard(const OperatorGuard &) = delete;
};
inline std::unique_lock<std::mutex> idle_lock(std::mutex &mutex) {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    const auto &held = held_operator_mutexes();
    // try_lock on a non-recursive mutex owned by this thread is undefined.
    if (std::find(held.begin(), held.end(), &mutex) == held.end()) lock.try_lock();
    return lock;
}
struct Registry {
    std::mutex mutex;
    std::unordered_map<uint64_t, metal_memory_stats> devices;
};
inline Registry &registry() { static Registry value; return value; }

inline uint64_t budget(id<MTLDevice> device) {
    uint64_t result = device.recommendedMaxWorkingSetSize;
    if (!result) result = device.maxBufferLength;
    // Optional byte budget for pressure validation and constrained machines.
    // It does not increase the device's recommended performance budget.
    const char *setting = std::getenv("VASP_METAL_MEMORY_BUDGET_BYTES");
    if (setting && *setting >= '0' && *setting <= '9') {
        char *end = nullptr; errno = 0;
        const auto value = std::strtoull(setting, &end, 10);
        if (!errno && end && !*end && value) result = std::min<uint64_t>(result, value);
    }
    return result;
}

inline void sample_locked(id<MTLDevice> device, metal_memory_stats &stats) {
    stats.current_device_allocated_bytes = device.currentAllocatedSize;
    stats.observed_device_peak_bytes = std::max(stats.observed_device_peak_bytes,
                                               stats.current_device_allocated_bytes);
    stats.device_budget_bytes = budget(device);
}

using PressureCallback = void (*)(const void *);

inline id<MTLBuffer> allocate(id<MTLDevice> device, size_t bytes,
                            MTLResourceOptions options,
                            PressureCallback pressure = nullptr,
                            const void *context = nullptr) NS_RETURNS_RETAINED {
    if (!device) return nil;
    auto &r = registry();
    std::unique_lock<std::mutex> lock(r.mutex);
    auto &stats = r.devices[device.registryID];
    ++stats.allocation_requests;
    sample_locked(device, stats);
    if (!can_allocate(stats.current_device_allocated_bytes, bytes,
                      stats.device_budget_bytes, device.maxBufferLength) && pressure) {
        // Never hold the allocator lock while trying operator locks. A caller
        // holding an active operator lock cannot evict that operator's buffers.
        ++stats.pressure_eviction_attempts;
        lock.unlock(); pressure(context); lock.lock();
        sample_locked(device, stats);
    }
    if (!can_allocate(stats.current_device_allocated_bytes, bytes,
                      stats.device_budget_bytes, device.maxBufferLength)) {
        ++stats.budget_rejections;
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:options];
    sample_locked(device, stats);
    if (!buffer) { ++stats.driver_allocation_failures; return nil; }
    // Driver alignment/bookkeeping can exceed the requested byte count. Keep
    // the transient observed peak visible, and release before reporting failure.
    if (stats.current_device_allocated_bytes > stats.device_budget_bytes) {
        ++stats.budget_rejections;
        buffer = nil;
        sample_locked(device, stats);
        return nil;
    }
    return buffer;
}

inline metal_memory_stats snapshot(id<MTLDevice> device) {
    if (!device) return {};
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto &stats = r.devices[device.registryID];
    sample_locked(device, stats);
    return stats;
}
}
