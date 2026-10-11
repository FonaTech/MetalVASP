#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

#include "double_single.hpp"
#include "metal_runner.hpp"
#include "metal_compile_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
constexpr char kShaderSource[] = R"METAL(
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct DS { float hi; float lo; };

inline DS ds_add(DS a, DS b) {
    float sum = a.hi + b.hi;
    float virtual_b = sum - a.hi;
    float error = ((a.hi - (sum - virtual_b)) + (b.hi - virtual_b)) + a.lo + b.lo;
    float hi = sum + error;
    return { hi, error - (hi - sum) };
}

inline DS ds_mul(DS a, DS b) {
    float product = a.hi * b.hi;
    float error = fma(a.hi, b.hi, -product)
                + a.hi * b.lo
                + a.lo * b.hi
                + a.lo * b.lo;
    float hi = product + error;
    return { hi, error - (hi - product) };
}

kernel void ds_chain(
    device const float2* lhs [[buffer(0)]],
    device const float2* rhs [[buffer(1)]],
    device float2* out [[buffer(2)]],
    constant uint& iterations [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    DS a = { lhs[index].x, lhs[index].y };
    DS b = { rhs[index].x, rhs[index].y };
    DS value = a;
    for (uint i = 0; i < iterations; ++i) {
        value = ds_add(ds_mul(value, a), b);
    }
    out[index] = float2(value.hi, value.lo);
}

kernel void float32_chain(
    device const float2* lhs [[buffer(0)]],
    device const float2* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& iterations [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    float a = lhs[index].x + lhs[index].y;
    float b = rhs[index].x + rhs[index].y;
    float value = a;
    for (uint i = 0; i < iterations; ++i) {
        value = fma(value, a, b);
    }
    out[index] = value;
}

struct GemmParams { uint m; uint n; uint k; uint repeats; };

kernel void ds_gemm(
    device const float2* a [[buffer(0)]],
    device const float2* b [[buffer(1)]],
    device float2* out [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint row = index / params.n;
    uint col = index % params.n;
    DS sum = { 0.0f, 0.0f };
    for (uint repeat = 0; repeat < params.repeats; ++repeat) {
        for (uint inner = 0; inner < params.k; ++inner) {
            DS left = { a[row * params.k + inner].x, a[row * params.k + inner].y };
            DS right = { b[inner * params.n + col].x, b[inner * params.n + col].y };
            sum = ds_add(sum, ds_mul(left, right));
        }
    }
    out[index] = float2(sum.hi, sum.lo);
}

kernel void float32_gemm(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint row = index / params.n;
    uint col = index % params.n;
    float sum = 0.0f;
    for (uint repeat = 0; repeat < params.repeats; ++repeat) {
        for (uint inner = 0; inner < params.k; ++inner) {
            sum = fma(a[row * params.k + inner], b[inner * params.n + col], sum);
        }
    }
    out[index] = sum;
}

)METAL";

constexpr char kFp64ProbeSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void native_fp64_probe(
    device const double* input [[buffer(0)]],
    device double* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    output[index] = input[index] * 2.0 + 1.0;
}
)METAL";

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, NSString *entry = @"ds_chain", std::string *language=nullptr) {
    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:kShaderSource];
    const bool fast=METAL_VASP_FP32_FAST_MATH && [entry hasPrefix:@"float32_"];
    auto compiled=metal_policy::compile(device,source,nil,fast);
    id<MTLLibrary> library=compiled.library;
    if(language) *language=metal_policy::language_name(compiled.language);
    if (library == nil) {
        throw std::runtime_error(compiled.error);
    }
    id<MTLFunction> function = [library newFunctionWithName:entry];
    if (function == nil) {
        throw std::runtime_error("Metal library does not contain ds_chain");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error([[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

id<MTLDevice> require_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil) return device;
    // Some macOS sessions do not expose a default device while still exposing
    // devices through the explicit enumeration API. Prefer the first low-power
    // device, then the first enumerated device, before failing with diagnostics.
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices.count > 0) {
        for (id<MTLDevice> candidate in devices) {
            if (![candidate isLowPower]) return candidate;
        }
        return devices.firstObject;
    }
    throw std::runtime_error("Metal runtime exposed neither a default device nor any enumerated devices; check the active macOS GUI session and GPU/Metal access");
}

CapabilityInfo query_capabilities_internal() {
    id<MTLDevice> device = require_device();
    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:kFp64ProbeSource];
    auto compiled=metal_policy::compile(device,source);
    id<MTLLibrary> library=compiled.library;
    if (!library) return {false, compiled.error};
    id<MTLFunction> function = [library newFunctionWithName:@"native_fp64_probe"];
    if (!function) return {false, "native_fp64_probe was not found"};
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return {false, [[error localizedDescription] UTF8String]};
    return {true, ""};
}

struct Timing {
    double wall_seconds;
    double gpu_seconds;
    bool has_gpu_time;
};

double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

Timing measure(id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> lhs,
               id<MTLBuffer> rhs,
               id<MTLBuffer> output,
               uint32_t iterations,
               uint64_t count) {
    std::vector<double> wall_samples;
    std::vector<double> gpu_samples;
    for (int repetition = 0; repetition < 6; ++repetition) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs offset:0 atIndex:0];
        [encoder setBuffer:rhs offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&iterations length:sizeof(iterations) atIndex:3];
        const NSUInteger thread_count = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        [encoder endEncoding];
        const auto start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        if ([command status] != MTLCommandBufferStatusCompleted) {
            NSString *detail = [[command error] localizedDescription] ?: @"unknown Metal error";
            throw std::runtime_error([detail UTF8String]);
        }
        const double wall = std::chrono::duration<double>(end - start).count();
        const double gpu = [command GPUEndTime] - [command GPUStartTime];
        if (repetition > 0) {
            wall_samples.push_back(wall);
            if (gpu > 0) gpu_samples.push_back(gpu);
        }
    }
    return {median(wall_samples), median(gpu_samples), !gpu_samples.empty()};
}

Timing measure_gemm(id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> lhs,
                    id<MTLBuffer> rhs,
                    id<MTLBuffer> output,
                    const uint32_t *params,
                    uint32_t params_size,
                    uint64_t count) {
    std::vector<double> wall_samples;
    std::vector<double> gpu_samples;
    for (int repetition = 0; repetition < 6; ++repetition) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs offset:0 atIndex:0];
        [encoder setBuffer:rhs offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:params length:params_size atIndex:3];
        const NSUInteger thread_count = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        [encoder endEncoding];
        const auto start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        const auto end = std::chrono::steady_clock::now();
        if ([command status] != MTLCommandBufferStatusCompleted) {
            NSString *detail = [[command error] localizedDescription] ?: @"unknown Metal error";
            throw std::runtime_error([detail UTF8String]);
        }
        const double wall = std::chrono::duration<double>(end - start).count();
        const double gpu = [command GPUEndTime] - [command GPUStartTime];
        if (repetition > 0) {
            wall_samples.push_back(wall);
            if (gpu > 0) gpu_samples.push_back(gpu);
        }
    }
    return {median(wall_samples), median(gpu_samples), !gpu_samples.empty()};
}

std::vector<simd_float2> make_inputs(uint64_t count, bool rhs) {
    std::vector<simd_float2> values(count);
    for (uint64_t index = 0; index < count; ++index) {
        const double value = rhs
            ? std::cos(static_cast<double>(index + 3) * 0.013) * 0.7
            : std::sin(static_cast<double>(index + 1) * 0.017) * 0.9;
        const DoubleSingle split = DoubleSingle::split(value);
        values[index] = simd_make_float2(split.hi, split.lo);
    }
    return values;
}

double reference_chain(simd_float2 lhs, simd_float2 rhs, uint32_t iterations) {
    const double a = static_cast<double>(lhs.x) + static_cast<double>(lhs.y);
    const double b = static_cast<double>(rhs.x) + static_cast<double>(rhs.y);
    double value = a;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        value = value * a + b;
    }
    return value;
}

} // namespace

CapabilityInfo query_capabilities() {
    return query_capabilities_internal();
}

GPUInfo query_gpu() {
    id<MTLDevice> device = require_device();
    GPUInfo info;
    info.name = [[device name] UTF8String];
    info.registry_id = [device registryID];
    info.recommended_working_set = [device recommendedMaxWorkingSetSize];
    info.max_threads_per_threadgroup = static_cast<uint32_t>(make_pipeline(device,@"ds_chain",&info.shader_language).maxTotalThreadsPerThreadgroup);
    info.compile_profile=metal_policy::profile();
    info.unified_memory = [device hasUnifiedMemory];
    info.metal4_family = false;
    info.ml_encoder_api_available = false;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        info.metal4_family = [device supportsFamily:MTLGPUFamilyMetal4];
        // Metal 4's ML command encoder is an API capability. It does not prove
        // that a VASP kernel will execute on ANE, nor that exact FP64/complex
        // operations are supported by that path.
        info.ml_encoder_api_available = info.metal4_family;
    }
#endif
    for (NSInteger family = 1; family <= 64; ++family) {
        if ([device supportsFamily:(MTLGPUFamily)(1000 + family)]) {
            info.apple_families.push_back(static_cast<int>(family));
        }
    }
    return info;
}

BenchmarkResult run_double_single_chain(uint64_t count, uint32_t iterations) {
    if (count == 0 || iterations == 0) {
        throw std::invalid_argument("count and iterations must be greater than zero");
    }
    id<MTLDevice> device = require_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("Could not create a Metal command queue");
    }
    id<MTLComputePipelineState> pipeline = make_pipeline(device);
    id<MTLComputePipelineState> float32_pipeline = make_pipeline(device, @"float32_chain");
    auto lhs = make_inputs(count, false);
    auto rhs = make_inputs(count, true);
    const NSUInteger buffer_length = static_cast<NSUInteger>(count * sizeof(simd_float2));
    id<MTLBuffer> lhs_buffer = [device newBufferWithBytes:lhs.data() length:buffer_length options:MTLResourceStorageModeShared];
    id<MTLBuffer> rhs_buffer = [device newBufferWithBytes:rhs.data() length:buffer_length options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [device newBufferWithLength:buffer_length options:MTLResourceStorageModeShared];
    id<MTLBuffer> float32_buffer = [device newBufferWithLength:static_cast<NSUInteger>(count * sizeof(float)) options:MTLResourceStorageModeShared];
    if (lhs_buffer == nil || rhs_buffer == nil || out_buffer == nil || float32_buffer == nil) {
        throw std::runtime_error("Could not allocate Metal buffers");
    }
    const Timing ds_timing = measure(queue, pipeline, lhs_buffer, rhs_buffer, out_buffer, iterations, count);
    const Timing float32_timing = measure(queue, float32_pipeline, lhs_buffer, rhs_buffer, float32_buffer, iterations, count);

    const auto *output = static_cast<const simd_float2 *>([out_buffer contents]);
    double max_absolute_error = 0;
    double max_relative_error = 0;
    bool finite = true;
    const uint64_t sample_stride = std::max<uint64_t>(1, count / 4096);
    for (uint64_t index = 0; index < count; ++index) {
        const double observed = static_cast<double>(output[index].x) + static_cast<double>(output[index].y);
        finite = finite && std::isfinite(observed);
        if (index % sample_stride != 0) continue;
        const double expected = reference_chain(lhs[index], rhs[index], iterations);
        const double absolute_error = std::abs(observed - expected);
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error = std::max(max_relative_error, absolute_error / std::max(std::abs(expected), 1e-300));
    }
    return {count, iterations, ds_timing.wall_seconds, ds_timing.gpu_seconds,
            float32_timing.wall_seconds, float32_timing.gpu_seconds,
            max_absolute_error, max_relative_error, ds_timing.has_gpu_time,
            float32_timing.has_gpu_time, finite};
}

GemmResult run_gemm(uint32_t m, uint32_t n, uint32_t k, uint32_t repeats) {
    if (m == 0 || n == 0 || k == 0 || repeats == 0) {
        throw std::invalid_argument("GEMM dimensions and repeats must be greater than zero");
    }
    id<MTLDevice> device = require_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) throw std::runtime_error("Could not create a Metal command queue");
    id<MTLComputePipelineState> ds_pipeline = make_pipeline(device, @"ds_gemm");
    id<MTLComputePipelineState> fp32_pipeline = make_pipeline(device, @"float32_gemm");

    const uint64_t a_count = static_cast<uint64_t>(m) * k;
    const uint64_t b_count = static_cast<uint64_t>(k) * n;
    const uint64_t c_count = static_cast<uint64_t>(m) * n;
    std::vector<simd_float2> ds_a(a_count), ds_b(b_count);
    std::vector<float> fp32_a(a_count), fp32_b(b_count);
    for (uint64_t index = 0; index < a_count; ++index) {
        const double value = std::sin(static_cast<double>(index + 1) * 0.001) * 0.25;
        const DoubleSingle split = DoubleSingle::split(value);
        ds_a[index] = simd_make_float2(split.hi, split.lo);
        fp32_a[index] = static_cast<float>(value);
    }
    for (uint64_t index = 0; index < b_count; ++index) {
        const double value = std::cos(static_cast<double>(index + 5) * 0.0007) * 0.25;
        const DoubleSingle split = DoubleSingle::split(value);
        ds_b[index] = simd_make_float2(split.hi, split.lo);
        fp32_b[index] = static_cast<float>(value);
    }
    const NSUInteger ds_a_bytes = static_cast<NSUInteger>(a_count * sizeof(simd_float2));
    const NSUInteger ds_b_bytes = static_cast<NSUInteger>(b_count * sizeof(simd_float2));
    const NSUInteger fp32_a_bytes = static_cast<NSUInteger>(a_count * sizeof(float));
    const NSUInteger fp32_b_bytes = static_cast<NSUInteger>(b_count * sizeof(float));
    const NSUInteger ds_c_bytes = static_cast<NSUInteger>(c_count * sizeof(simd_float2));
    const NSUInteger fp32_c_bytes = static_cast<NSUInteger>(c_count * sizeof(float));
    id<MTLBuffer> ds_a_buffer = [device newBufferWithBytes:ds_a.data() length:ds_a_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> ds_b_buffer = [device newBufferWithBytes:ds_b.data() length:ds_b_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> fp32_a_buffer = [device newBufferWithBytes:fp32_a.data() length:fp32_a_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> fp32_b_buffer = [device newBufferWithBytes:fp32_b.data() length:fp32_b_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> ds_c_buffer = [device newBufferWithLength:ds_c_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> fp32_c_buffer = [device newBufferWithLength:fp32_c_bytes options:MTLResourceStorageModeShared];
    if (!ds_a_buffer || !ds_b_buffer || !fp32_a_buffer || !fp32_b_buffer || !ds_c_buffer || !fp32_c_buffer) {
        throw std::runtime_error("Could not allocate GEMM Metal buffers");
    }
    const uint32_t params[4] = {m, n, k, repeats};
    const uint64_t output_count = c_count;
    const Timing ds_timing = measure_gemm(queue, ds_pipeline, ds_a_buffer, ds_b_buffer, ds_c_buffer,
                                          params, sizeof(params), output_count);
    const Timing fp32_timing = measure_gemm(queue, fp32_pipeline, fp32_a_buffer, fp32_b_buffer, fp32_c_buffer,
                                            params, sizeof(params), output_count);

    const auto *ds_output = static_cast<const simd_float2 *>([ds_c_buffer contents]);
    double max_absolute_error = 0;
    double max_relative_error = 0;
    bool finite = true;
    const uint64_t sample_stride = std::max<uint64_t>(1, c_count / 4096);
    for (uint64_t index = 0; index < c_count; ++index) {
        const double observed = static_cast<double>(ds_output[index].x) + static_cast<double>(ds_output[index].y);
        finite = finite && std::isfinite(observed);
        if (index % sample_stride != 0) continue;
        const uint32_t row = static_cast<uint32_t>(index / n);
        const uint32_t col = static_cast<uint32_t>(index % n);
        double expected = 0;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            for (uint32_t inner = 0; inner < k; ++inner) {
                expected += (static_cast<double>(ds_a[static_cast<uint64_t>(row) * k + inner].x)
                           + static_cast<double>(ds_a[static_cast<uint64_t>(row) * k + inner].y))
                          * (static_cast<double>(ds_b[static_cast<uint64_t>(inner) * n + col].x)
                           + static_cast<double>(ds_b[static_cast<uint64_t>(inner) * n + col].y));
            }
        }
        const double absolute_error = std::abs(observed - expected);
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error = std::max(max_relative_error, absolute_error / std::max(std::abs(expected), 1e-300));
    }
    const uint64_t mathematical_flops = 2ull * static_cast<uint64_t>(m) * n * k * repeats;
    const double ds_tflops = ds_timing.has_gpu_time && ds_timing.gpu_seconds > 0
        ? static_cast<double>(mathematical_flops) / ds_timing.gpu_seconds / 1e12 : 0;
    const double fp32_tflops = fp32_timing.has_gpu_time && fp32_timing.gpu_seconds > 0
        ? static_cast<double>(mathematical_flops) / fp32_timing.gpu_seconds / 1e12 : 0;
    return {m, n, k, repeats, mathematical_flops,
            ds_timing.wall_seconds, ds_timing.gpu_seconds,
            fp32_timing.wall_seconds, fp32_timing.gpu_seconds,
            ds_tflops, fp32_tflops, max_absolute_error, max_relative_error,
            ds_timing.has_gpu_time, fp32_timing.has_gpu_time, finite};
}
