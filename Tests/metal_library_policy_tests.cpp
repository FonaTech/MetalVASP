#include "metal_library_policy.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdio>
int main() {
    using namespace metal_library;
    Metadata good;Expectations e;
    good.compilation_success=good.assembly_verified=true;good.status="offline-compilation-succeeded";
    good.shader_hash=e.shader_hash=std::string(64,'a');good.library_hash=e.library_hash=std::string(64,'b');
    good.language="4.0";e.language_ceiling="4.0";good.deployment_target="26.0";e.os_version={{26,1,0}};
    good.device_tuned=e.device_tuned=1;good.metal4=e.metal4=1;
    for(auto name:part_names) {good.symbols.push_back(name);good.part_hashes.emplace_back(64,'c');e.part_hashes.emplace_back(64,'c');}
    for(auto flag:safe_flags) good.flags.push_back(flag);
    good.commands_success=true;good.compiler_arguments={"xcrun","metal","-std=metal4.0","-mmacosx-version-min=26.0",good.flags[0],good.flags[1],good.flags[2],"-DMETAL_VASP_DEVICE_TUNED=1","-DMETAL_VASP_METAL4=1","-c","source.metal","-o","output.air"};
    good.kernels=e.kernels={"ds_zgemm","ds_zgemm_tiled"};assert(validate(good,e).empty());
    auto bad=good;bad.flags[0]="-fmetal-math-mode=fast";assert(validate(bad,e)=="unsafe-compiler-flags");
    bad=good;bad.flags.pop_back();assert(validate(bad,e)=="unsafe-compiler-flags");
    bad=good;bad.shader_hash[0]='b';assert(validate(bad,e)=="shader-sha256-mismatch");
    bad=good;bad.library_hash[0]='a';assert(validate(bad,e)=="metallib-sha256-mismatch");
    bad=good;bad.part_hashes[4][0]='d';assert(validate(bad,e)=="source-part-mismatch:kCyclicShader");
    bad=good;bad.symbols.pop_back();assert(validate(bad,e)=="source-parts-incomplete");
    bad=good;bad.device_tuned=0;assert(validate(bad,e)=="profile-macros-mismatch");
    bad=good;bad.metal4=0;assert(validate(bad,e)=="profile-macros-mismatch");
    auto limited=e;limited.language_ceiling="3.2";assert(validate(good,limited)=="language-exceeds-runtime-ceiling");
    bad=good;bad.language="4.1";assert(validate(bad,e)=="language-exceeds-runtime-ceiling");
    bad=good;bad.deployment_target="";assert(validate(bad,e)=="deployment-target-missing-or-invalid");
    bad=good;bad.deployment_target="13.0";assert(validate(bad,e)=="deployment-target-below-language-minimum");
    bad=good;bad.deployment_target="26.2";assert(validate(bad,e)=="deployment-target-exceeds-runtime-os");
    bad=good;bad.compilation_success=false;assert(validate(bad,e)=="compilation-not-successful");
    bad=good;bad.assembly_verified=false;assert(validate(bad,e)=="runtime-source-assembly-unverified");
    bad=good;bad.commands_success=false;assert(validate(bad,e)=="compiler-provenance-invalid");
    bad=good;bad.compiler_arguments[3]="-mmacosx-version-min=13.0";assert(validate(bad,e)=="compiler-provenance-invalid");
    bad=good;bad.compiler_arguments[4]="-fmetal-math-mode=fast";assert(validate(bad,e)=="compiler-provenance-invalid");
    bad=good;bad.kernels.push_back(bad.kernels[0]);assert(validate(bad,e)=="kernel-list-mismatch");
    bad=good;bad.kernels.pop_back();assert(validate(bad,e)=="kernel-list-mismatch");
    good.language="2.3";good.deployment_target="13.0";good.device_tuned=e.device_tuned=0;good.metal4=e.metal4=0;e.language_ceiling="2.4";e.os_version={{13,0,0}};
    good.compiler_arguments[2]="-std=macos-metal2.3";good.compiler_arguments[3]="-mmacosx-version-min=13.0";good.compiler_arguments[7]="-DMETAL_VASP_DEVICE_TUNED=0";good.compiler_arguments[8]="-DMETAL_VASP_METAL4=0";assert(validate(good,e).empty());
    std::array<int,3> v{},expected_v{{13,0,1}};assert(version("13.0.1",v)&&v==expected_v);
    assert(!version("13",v)&&!version("13.0.",v)&&!version("13.0.0.1",v)&&!version("-13.0",v));
    std::puts("offline library metadata policy gates passed; CPU only");
}
