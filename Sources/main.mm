#include "metal_runner.hpp"
#include "metal_compile_policy.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

static std::string json_escape(const std::string &value) {
    std::string escaped;
    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

int main(int argc, char **argv) {
    try {
        const uint64_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 262144;
        const uint32_t iterations = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 128;
        const uint32_t gemm_m = argc > 3 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 256;
        const uint32_t gemm_n = argc > 4 ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 256;
        const uint32_t gemm_k = argc > 5 ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 256;
        const uint32_t gemm_repeats = argc > 6 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 16;
        const GPUInfo device = query_gpu();
        const CapabilityInfo capabilities = query_capabilities();
        const BenchmarkResult result = run_double_single_chain(count, iterations);
        const GemmResult gemm = run_gemm(gemm_m, gemm_n, gemm_k, gemm_repeats);
        const bool pass = result.finite && gemm.finite && result.max_absolute_error < 1e-10 && gemm.ds_max_absolute_error < 1e-10;

        std::cout << std::setprecision(12)
                  << "{\n"
                  << "  \"compilation\": {\"profile\": \"" << device.compile_profile
                  << "\", \"shader_language\": \"" << device.shader_language
                  << "\", \"double_single_math\": \"safe-precise\", \"fp32_fast_math\": "
                  << (METAL_VASP_FP32_FAST_MATH ? "true" : "false") << "},\n"
                  << "  \"device\": {\"name\": \"" << device.name << "\", \"registry_id\": " << device.registry_id
                  << ", \"recommended_working_set_bytes\": " << device.recommended_working_set
                  << ", \"max_threads_per_threadgroup\": " << device.max_threads_per_threadgroup
                  << ", \"unified_memory\": " << (device.unified_memory ? "true" : "false")
                  << ", \"metal4_family\": " << (device.metal4_family ? "true" : "false")
                  << ", \"ml_encoder_api_available\": " << (device.ml_encoder_api_available ? "true" : "false")
                  << ", \"apple_gpu_families\": [";
        for (size_t index = 0; index < device.apple_families.size(); ++index) {
            if (index) std::cout << ", ";
            std::cout << device.apple_families[index];
        }
        std::cout << "]},\n"
                  << "  \"capabilities\": {\"native_fp64_shader_compiles\": "
                  << (capabilities.native_fp64_shader_compiles ? "true" : "false")
                  << ", \"native_fp64_tflops\": null"
                  << ", \"native_fp64_error\": \"" << json_escape(capabilities.native_fp64_error) << "\"},\n"
                  << "  \"result\": {\"count\": " << result.count << ", \"iterations\": " << result.iterations
                  << ", \"wall_seconds\": " << result.wall_seconds
                  << ", \"gpu_seconds\": " << (result.has_gpu_time ? result.gpu_seconds : 0)
                  << ", \"gpu_time_available\": " << (result.has_gpu_time ? "true" : "false")
                  << ", \"float32_wall_seconds\": " << result.float32_wall_seconds
                  << ", \"float32_gpu_seconds\": " << (result.has_float32_gpu_time ? result.float32_gpu_seconds : 0)
                  << ", \"float32_gpu_time_available\": " << (result.has_float32_gpu_time ? "true" : "false")
                  << ", \"double_single_vs_float32_gpu_ratio\": "
                  << (result.has_gpu_time && result.has_float32_gpu_time && result.float32_gpu_seconds > 0
                          ? result.float32_gpu_seconds / result.gpu_seconds : 0)
                  << ", \"max_absolute_error\": " << result.max_absolute_error
                  << ", \"max_relative_error\": " << result.max_relative_error
                  << ", \"finite\": " << (result.finite ? "true" : "false")
                  << ", \"pass\": " << (pass ? "true" : "false") << "},\n"
                  << "  \"gemm\": {\"m\": " << gemm.m << ", \"n\": " << gemm.n << ", \"k\": " << gemm.k
                  << ", \"repeats\": " << gemm.repeats << ", \"mathematical_flops\": " << gemm.mathematical_flops
                  << ", \"ds_wall_seconds\": " << gemm.ds_wall_seconds << ", \"ds_gpu_seconds\": " << gemm.ds_gpu_seconds
                  << ", \"fp32_wall_seconds\": " << gemm.fp32_wall_seconds << ", \"fp32_gpu_seconds\": " << gemm.fp32_gpu_seconds
                  << ", \"ds_fp64_equivalent_tflops\": " << gemm.ds_fp64_equivalent_tflops
                  << ", \"fp32_tflops\": " << gemm.fp32_tflops
                  << ", \"ds_max_absolute_error\": " << gemm.ds_max_absolute_error
                  << ", \"ds_max_relative_error\": " << gemm.ds_max_relative_error
                  << ", \"ds_gpu_time_available\": " << (gemm.ds_gpu_time_available ? "true" : "false")
                  << ", \"fp32_gpu_time_available\": " << (gemm.fp32_gpu_time_available ? "true" : "false")
                  << ", \"finite\": " << (gemm.finite ? "true" : "false") << "}\n}\n";
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "metal-vasp-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
