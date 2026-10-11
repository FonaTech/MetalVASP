#pragma once
#include <cerrno>
#include <cstdlib>
#include <string>

namespace metal_stats {
struct MpiIdentity {
    long long rank=0, size=1;
    bool known=false;
};
inline bool nonnegative(const char *text, long long &value) {
    if(!text || !*text) return false;
    for(const char *p=text;*p;++p) if(*p<'0' || *p>'9') return false;
    char *end=nullptr; errno=0; value=std::strtoll(text,&end,10);
    return errno==0 && end && !*end && value>=0;
}
inline MpiIdentity mpi_identity() {
    const char *pairs[][2]={{"OMPI_COMM_WORLD_RANK","OMPI_COMM_WORLD_SIZE"},
        {"PMI_RANK","PMI_SIZE"},{"PMIX_RANK","PMIX_UNIV_SIZE"},
        {"SLURM_PROCID","SLURM_NTASKS"}};
    for(const auto &pair:pairs) {
        MpiIdentity result;
        if(!nonnegative(std::getenv(pair[0]),result.rank)) continue;
        // An absent size cannot establish a serial job. Keep distinct files
        // whenever a known global rank exists without a valid world size.
        if(!nonnegative(std::getenv(pair[1]),result.size) || result.size==0) result.size=0;
        if(result.size>0 && result.rank>=result.size) continue;
        result.known=true; return result;
    }
    return {};
}
inline std::string path(const char *requested, const MpiIdentity &mpi) {
    std::string result=requested && *requested ? requested : "VASP_METAL_STATS.json";
    const std::string rank=std::to_string(mpi.rank);
    bool expanded=false;
    for(size_t at=0;(at=result.find("%r",at))!=std::string::npos;) {
        result.replace(at,2,rank); at+=rank.size(); expanded=true;
    }
    if(!expanded && mpi.known && mpi.size!=1) {
        const std::string suffix=".rank"+rank;
        const size_t slash=result.find_last_of("/\\"), dot=result.find_last_of('.');
        const size_t first=slash==std::string::npos ? 0 : slash+1;
        if(dot!=std::string::npos && dot>first) result.insert(dot,suffix);
        else result+=suffix;
    }
    return result;
}
inline std::string json_string(const char *text) {
    std::string result="\"";
    if(text) for(const unsigned char *p=reinterpret_cast<const unsigned char *>(text);*p;++p) {
        switch(*p) {
        case '"': result+="\\\""; break;
        case '\\': result+="\\\\"; break;
        case '\n': result+="\\n"; break;
        case '\r': result+="\\r"; break;
        case '\t': result+="\\t"; break;
        default:
            if(*p<32) { const char digits[]="0123456789abcdef"; result+="\\u00"; result+=digits[*p>>4]; result+=digits[*p&15]; }
            else result+=char(*p);
        }
    }
    result+='"'; return result;
}
}
