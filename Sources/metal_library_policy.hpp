#pragma once
#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <vector>

namespace metal_library {
inline const std::array<const char *,9> part_names = {{
    "kBackendShader", "kResidentShader", "kLobpcgShader", "kSubspaceShader",
    "kCyclicShader", "hamiltonian_shader", "orthogonal_shader", "wave_owner_shader", "nonlocal_shader"
}};
inline const std::array<const char *,3> safe_flags = {{
    "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-ffp-contract=off"
}};
struct Metadata {
    bool compilation_success=false, assembly_verified=false, commands_success=false;
    std::string status, shader_hash, library_hash, language, deployment_target;
    int device_tuned=-1, metal4=-1;
    std::vector<std::string> flags, symbols, part_hashes, kernels, compiler_arguments;
};
struct Expectations {
    std::string shader_hash, library_hash, language_ceiling;
    int device_tuned=0, metal4=0;
    std::array<int,3> os_version{{0,0,0}};
    std::vector<std::string> part_hashes, kernels;
};
inline bool digest(const std::string &s) {
    return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c) {
        return (c>='0'&&c<='9')||(c>='a'&&c<='f');
    });
}
inline unsigned language_code(const std::string &s) {
    if(s=="2.3") return 203; if(s=="2.4") return 204;
    if(s=="3.0") return 300; if(s=="3.1") return 301;
    if(s=="3.2") return 302; if(s=="4.0") return 400;
    return 0;
}
inline bool version(const std::string &s,std::array<int,3> &out) {
    out={{0,0,0}}; unsigned fields=0; size_t pos=0;
    while(pos<s.size() && fields<3) {
        if(!std::isdigit(static_cast<unsigned char>(s[pos]))) return false;
        unsigned value=0;
        while(pos<s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
            value=value*10+unsigned(s[pos++]-'0'); if(value>9999) return false;
        }
        out[fields++]=int(value);
        if(pos==s.size()) return fields>=2;
        if(s[pos++]!='.' || pos==s.size()) return false;
    }
    return false;
}
inline std::string validate(const Metadata &m,const Expectations &e) {
    if(!m.compilation_success || m.status!="offline-compilation-succeeded") return "compilation-not-successful";
    if(!m.assembly_verified) return "runtime-source-assembly-unverified";
    if(!digest(m.shader_hash) || m.shader_hash!=e.shader_hash) return "shader-sha256-mismatch";
    if(!digest(m.library_hash) || m.library_hash!=e.library_hash) return "metallib-sha256-mismatch";
    if(m.symbols.size()!=part_names.size() || m.part_hashes.size()!=part_names.size() || e.part_hashes.size()!=part_names.size()) return "source-parts-incomplete";
    for(size_t i=0;i<part_names.size();++i)
        if(m.symbols[i]!=part_names[i] || !digest(m.part_hashes[i]) || m.part_hashes[i]!=e.part_hashes[i]) return "source-part-mismatch:"+std::string(part_names[i]);
    if(m.flags.size()!=safe_flags.size()) return "unsafe-compiler-flags";
    for(auto flag:safe_flags) if(std::count(m.flags.begin(),m.flags.end(),flag)!=1) return "unsafe-compiler-flags";
    if(m.device_tuned!=e.device_tuned || m.metal4!=e.metal4) return "profile-macros-mismatch";
    const unsigned language=language_code(m.language),ceiling=language_code(e.language_ceiling);
    if(!language || !ceiling || language>ceiling) return "language-exceeds-runtime-ceiling";
    std::array<int,3> deployment{};
    if(!version(m.deployment_target,deployment)) return "deployment-target-missing-or-invalid";
    const int minimum=language==400?26:language==302?15:language==301?14:13;
    if(deployment[0]<minimum) return "deployment-target-below-language-minimum";
    if(deployment>e.os_version) return "deployment-target-exceeds-runtime-os";
    const auto &args=m.compiler_arguments;
    const std::string standard=(language<300?"macos-metal":"metal")+m.language;
    if(!m.commands_success || args.size()!=13 || args[0]!="xcrun" || args[1]!="metal" ||
       args[2]!="-std="+standard || args[3]!="-mmacosx-version-min="+m.deployment_target ||
       args[7]!="-DMETAL_VASP_DEVICE_TUNED="+std::to_string(m.device_tuned) ||
       args[8]!="-DMETAL_VASP_METAL4="+std::to_string(m.metal4) ||
       args[9]!="-c" || args[10].empty() || args[11]!="-o" || args[12].empty() ||
       !std::equal(m.flags.begin(),m.flags.end(),args.begin()+4)) return "compiler-provenance-invalid";
    auto actual=m.kernels,expected=e.kernels;
    std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
    if(actual.empty() || actual!=expected || std::adjacent_find(actual.begin(),actual.end())!=actual.end()) return "kernel-list-mismatch";
    return {};
}
}
