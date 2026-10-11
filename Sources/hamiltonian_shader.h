#pragma once
// Includes no VASP code; all arrays enter via the external ISO_C_BINDING ABI.
static const char *hamiltonian_shader=R"METAL(
struct HamParams {uint total,npl,bands,nions,npro,lmdim,ntypes,qld,qchannels; uint real_potential;};
struct HamSite {uint ion,type,base,count,l;};
kernel void ds_ham_scatter(device float4* grid [[buffer(0)]],device const float4* cw [[buffer(1)]],
 device const uint* index [[buffer(2)]],constant HamParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.npl*p.bands) return;uint band=i/p.npl,g=i%p.npl;
 grid[band*p.total+index[g]]=cw[i];
}
kernel void ds_ham_local(device float4* grid [[buffer(0)]],device const float4* potential [[buffer(1)]],
    constant HamParams& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.total*p.bands) return;
    DSC value=load_dsc(grid,i),v=load_dsc(potential,i%p.total);
    value=dsc_mul(value,v);DS inv={1.0f/float(p.total),0.0f};
    // The host supplies 1/N as DS in the last potential slot.
    float4 scale=potential[p.total];inv={scale.x,scale.y};
    grid[i]=store_dsc((DSC){ds_mul(value.real,inv),ds_mul(value.imag,inv)});
}
kernel void ds_ham_onsite(device const float4* d [[buffer(0)]],device const float4* q [[buffer(1)]],
    device const float4* projection [[buffer(2)]],device const float2* eig [[buffer(3)]],
    device const HamSite* site [[buffer(4)]],device float4* out [[buffer(5)]],constant HamParams& p [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.npro*p.bands) return;uint band=i/p.npro,j=i%p.npro;HamSite s=site[j];
    DS e={eig[band].x,eig[band].y};DSC value={{0,0},{0,0}};
    for(uint l=0;l<s.count;++l) {
        uint at=l+s.l*p.lmdim+s.ion*p.lmdim*p.lmdim;
        DSC dd=load_dsc(d,at),qq=load_dsc(q,at);
        dd=dsc_add(dd,(DSC){ds_neg(ds_mul(e,qq.real)),ds_neg(ds_mul(e,qq.imag))});
        value=dsc_add(value,dsc_mul(dd,load_dsc(projection,band*p.npro+s.base+l)));
    }
    out[i]=store_dsc(value);
}
kernel void ds_ham_finish(device const float4* grid [[buffer(0)]],device const uint* index [[buffer(1)]],
    device const float4* cw [[buffer(2)]],device const float2* kinetic [[buffer(3)]],device const float2* eig [[buffer(4)]],
    device const float4* onsite [[buffer(5)]],device const float2* radial [[buffer(6)]],device const float4* cq [[buffer(7)]],
    device const float4* phase [[buffer(8)]],device const HamSite* site [[buffer(9)]],device float4* out [[buffer(10)]],
    constant HamParams& p [[buffer(11)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.npl*p.bands) return;uint band=i/p.npl,g=i%p.npl;
    DSC value=load_dsc(grid,band*p.total+index[g]);
    DS ke={kinetic[g].x,kinetic[g].y},e={eig[band].x,eig[band].y};
    DS weight=ds_add(ke,ds_neg(e));DSC w=load_dsc(cw,i);
    value=dsc_add(value,(DSC){ds_mul(w.real,weight),ds_mul(w.imag,weight)});
    // One thread owns each G coefficient, so no FP atomics or ion races.
    for(uint j=0;j<p.npro;++j) {
        HamSite s=site[j];uint qindex=g+p.qld*(s.l+p.qchannels*s.type);
        DS r={radial[qindex].x,radial[qindex].y};
        DSC f=load_dsc(cq,s.l+p.qchannels*s.type);f.imag=ds_neg(f.imag);
        DSC c=dsc_mul(load_dsc(onsite,band*p.npro+j),f);
        DSC ph=load_dsc(phase,g+p.npl*s.ion);ph.imag=ds_neg(ph.imag);
        c=dsc_mul(c,ph);value=dsc_add(value,(DSC){ds_mul(c.real,r),ds_mul(c.imag,r)});
    }
    out[i]=store_dsc(value);
}
)METAL";
