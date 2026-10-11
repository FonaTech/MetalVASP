#pragma once
// Original kernels consume explicitly imported compact VASP/PAW array data.
static const char *wave_owner_shader=R"METAL(
struct WaveOwnerParams { uint npl,bands,npro,nions,lmdim,qchannels,use_overlap; };
struct WaveOwnerSite { uint ion,type,base,count,l; };
kernel void ds_wave_precondition(device float4 *wave [[buffer(0)]],device const float2 *preconditioner [[buffer(1)]],
 constant WaveOwnerParams &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.npl*p.bands) return;
 wave[i]=store_dsc(dsc_scale_ds(load_dsc(wave,i),preconditioner[i]));
}
kernel void ds_wave_project(device const float4 *wave [[buffer(0)]],device const float2 *radial [[buffer(1)]],
 device const float4 *cq [[buffer(2)]],device const float4 *phase [[buffer(3)]],device const WaveOwnerSite *site [[buffer(4)]],
 device float4 *projection [[buffer(5)]],constant WaveOwnerParams &p [[buffer(6)]],
 uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
 threadgroup DSC partial[256];
 uint band=group/p.npro,j=group%p.npro;WaveOwnerSite s=site[j];
 DSC sum={{0,0},{0,0}};
 for(uint g=tid;g<p.npl;g+=256u) {
   DSC value=dsc_mul(load_dsc(phase,g+p.npl*s.ion),load_dsc(wave,g+p.npl*band));
   uint at=g+p.npl*(s.l+p.qchannels*s.type);
   sum=dsc_add(sum,dsc_scale_ds(value,radial[at]));
 }
 partial[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=128u;width; width>>=1u) {
   if(tid<width) partial[tid]=dsc_add(partial[tid],partial[tid+width]);
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(!tid) projection[group]=store_dsc(dsc_mul(partial[0],load_dsc(cq,s.l+p.qchannels*s.type)));
}
kernel void ds_wave_metric(device const float4 *projection [[buffer(0)]],device const float4 *metric [[buffer(1)]],
 device const WaveOwnerSite *site [[buffer(2)]],device float4 *metric_projection [[buffer(3)]],
 constant WaveOwnerParams &p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.npro*p.bands) return;uint band=i/p.npro,j=i%p.npro;WaveOwnerSite s=site[j];
 DSC sum={{0,0},{0,0}};
 if(p.use_overlap) for(uint k=0;k<s.count;++k)
   sum=dsc_add(sum,dsc_mul(load_dsc(metric,k+s.l*p.lmdim+s.ion*p.lmdim*p.lmdim),
                           load_dsc(projection,band*p.npro+s.base+k)));
 metric_projection[i]=store_dsc(sum);
}
kernel void ds_wave_norm(device const float4 *wave [[buffer(0)]],device const float4 *projection [[buffer(1)]],
 device const float4 *metric_projection [[buffer(2)]],device float2 *scale [[buffer(3)]],device uint *status [[buffer(4)]],
 constant WaveOwnerParams &p [[buffer(5)]],uint tid [[thread_index_in_threadgroup]],uint band [[threadgroup_position_in_grid]]) {
 threadgroup DS partial[256];DS sum={0,0};
 for(uint g=tid;g<p.npl;g+=256u) {DSC z=load_dsc(wave,g+band*p.npl);sum=ds_add(sum,dsc_mul(dsc_conj(z),z).real);}
 if(p.use_overlap) for(uint j=tid;j<p.npro;j+=256u)
   sum=ds_add(sum,dsc_mul(dsc_conj(load_dsc(projection,j+band*p.npro)),load_dsc(metric_projection,j+band*p.npro)).real);
 partial[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=128u;width;width>>=1u) {
   if(tid<width) partial[tid]=ds_add(partial[tid],partial[tid+width]);
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(!tid) {
   DS norm=partial[0];
   if(!isfinite(norm.hi) || !isfinite(norm.lo) || norm.hi<=0.0f) {status[band]=1u;scale[band]=float2(0);}
   else {DS inv=ds_div(DS{1,0},ds_sqrt(norm));scale[band]=float2(inv.hi,inv.lo);status[band]=0u;}
 }
}
kernel void ds_wave_normalize(device float4 *wave [[buffer(0)]],device float4 *projection [[buffer(1)]],
 device float4 *metric_projection [[buffer(2)]],device const float2 *scale [[buffer(3)]],
 constant WaveOwnerParams &p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 uint width=p.npl+2u*p.npro;if(i>=width*p.bands) return;
 uint band=i/width,j=i%width;float2 s=scale[band];
 if(j<p.npl) {uint at=j+band*p.npl;wave[at]=store_dsc(dsc_scale_ds(load_dsc(wave,at),s));}
 else if(j<p.npl+p.npro) {uint at=j-p.npl+band*p.npro;projection[at]=store_dsc(dsc_scale_ds(load_dsc(projection,at),s));}
 else {uint at=j-p.npl-p.npro+band*p.npro;metric_projection[at]=store_dsc(dsc_scale_ds(load_dsc(metric_projection,at),s));}
}
)METAL";
