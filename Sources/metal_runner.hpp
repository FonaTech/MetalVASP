#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GPUInfo {
    std::string compile_profile;
    std::string shader_language;
    std::string name;
    uint64_t registry_id;
    uint64_t recommended_working_set;
    uint32_t max_threads_per_threadgroup;
    bool unified_memory;
    bool metal4_family;
    bool ml_encoder_api_available;
    std::vector<int> apple_families;
};

struct BenchmarkResult {
    uint64_t count;
    uint32_t iterations;
    double wall_seconds;
    double gpu_seconds;
    double float32_wall_seconds;
    double float32_gpu_seconds;
    double max_absolute_error;
    double max_relative_error;
    bool has_gpu_time;
    bool has_float32_gpu_time;
    bool finite;
};

struct CapabilityInfo {
    bool native_fp64_shader_compiles;
    std::string native_fp64_error;
};

struct GemmResult {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t repeats;
    uint64_t mathematical_flops;
    double ds_wall_seconds;
    double ds_gpu_seconds;
    double fp32_wall_seconds;
    double fp32_gpu_seconds;
    double ds_fp64_equivalent_tflops;
    double fp32_tflops;
    double ds_max_absolute_error;
    double ds_max_relative_error;
    bool ds_gpu_time_available;
    bool fp32_gpu_time_available;
    bool finite;
};

GPUInfo query_gpu();
CapabilityInfo query_capabilities();
BenchmarkResult run_double_single_chain(uint64_t count, uint32_t iterations);
GemmResult run_gemm(uint32_t m, uint32_t n, uint32_t k, uint32_t repeats);
