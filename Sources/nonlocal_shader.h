#pragma once
static const char *nonlocal_shader = R"METAL(
struct NonlocalSite { ulong ion,type,l; };
struct NonlocalParams { ulong ng,ns,nb,nq,wave_stride,wave_g,wave_band; uint channels,first_g; float4 multiplier; };
kernel void ds_nonlocal_project(device const float4 *wave [[buffer(0)]],
 device const float2 *radial [[buffer(1)]],device const float4 *phase [[buffer(2)]],
 device const float4 *cq [[buffer(3)]],device const NonlocalSite *site [[buffer(4)]],
 device const float2 *gradient [[buffer(5)]],device float4 *result [[buffer(6)]],
 constant NonlocalParams &p [[buffer(7)]],uint tid [[thread_index_in_threadgroup]],
 uint group [[threadgroup_position_in_grid]]) {
 threadgroup DSC partial[4][256];
 const ulong band=ulong(group)/p.ns,j=ulong(group)%p.ns;
 const NonlocalSite s=site[j];
 DSC v[4]={{{0,0},{0,0}},{{0,0},{0,0}},{{0,0},{0,0}},{{0,0},{0,0}}};
 for(ulong g=tid;g<p.ng;g+=256ul) {
   DSC z=dsc_scale_ds(dsc_mul(load_dsc(phase,g+p.ng*s.ion),
                    load_dsc(wave,g+p.wave_g+p.wave_stride*(band+p.wave_band))),
                    radial[g+p.ng*(s.l+p.nq*s.type)]);
   v[0]=dsc_add(v[0],z);
   if(p.channels==4u) for(uint c=1;c<4u;++c)
     v[c]=dsc_add(v[c],dsc_scale_ds(z,gradient[g+p.ng*ulong(c-1u)]));
 }
 for(uint c=0;c<p.channels;++c) partial[c][tid]=v[c];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=128u;width;width>>=1u) {
   if(tid<width) for(uint c=0;c<p.channels;++c)
     partial[c][tid]=dsc_add(partial[c][tid],partial[c][tid+width]);
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(!tid) {
   const DSC q=load_dsc(cq,s.l+p.nq*s.type);
   const DSC m={{p.multiplier.x,p.multiplier.y},{p.multiplier.z,p.multiplier.w}};
   for(uint c=0;c<p.channels;++c) {
     ulong at=j+p.ns*(band+p.nb*ulong(c));
     DSC z=dsc_mul(partial[c][0],c ? dsc_mul(q,m):q);
     if(!p.first_g) z=dsc_add(load_dsc(result,at),z);
     result[at]=store_dsc(z);
   }
 }
}
)METAL";
