#pragma once
static const char *orthogonal_shader=R"METAL(
struct OrthParams {uint rows,cols,npl,npro;};
kernel void ds_orth_overlap(device const float4* wave_a [[buffer(0)]],device const float4* wave_b [[buffer(1)]],
 device const float4* proj_a [[buffer(2)]],device const float4* proj_b [[buffer(3)]],device float4* overlap [[buffer(4)]],
 constant OrthParams& p [[buffer(5)]],uint2 grid [[thread_position_in_grid]]) {
 if(grid.x>=p.rows || grid.y>=p.cols)return;uint row=grid.x,col=grid.y;ulong i=ulong(row)+ulong(col)*p.rows;DSC sum=load_dsc(overlap,i);
 for(uint k=0;k<p.npl;++k) {DSC a=load_dsc(wave_a,k+ulong(row)*p.npl);a.imag=ds_neg(a.imag);sum=dsc_add(sum,dsc_mul(a,load_dsc(wave_b,k+ulong(col)*p.npl)));}
 for(uint k=0;k<p.npro;++k) {DSC a=load_dsc(proj_a,k+ulong(row)*p.npro);a.imag=ds_neg(a.imag);sum=dsc_add(sum,dsc_mul(a,load_dsc(proj_b,k+ulong(col)*p.npro)));}
 overlap[i]=store_dsc(sum);
}
// One dot product per group. Adjacent lanes read adjacent plane waves; the
// original scalar kernel remains available for small inputs and controls.
kernel void ds_orth_overlap_group(device const float4* wave_a [[buffer(0)]],device const float4* wave_b [[buffer(1)]],
 device const float4* proj_a [[buffer(2)]],device const float4* proj_b [[buffer(3)]],device float4* overlap [[buffer(4)]],
 constant OrthParams& p [[buffer(5)]],uint3 group [[threadgroup_position_in_grid]],
 uint3 local [[thread_position_in_threadgroup]],uint3 group_size [[threads_per_threadgroup]]) {
 if(group.x>=p.rows || group.y>=p.cols)return;
 uint lanes=group_size.x,lane=local.x;
 uint row=group.x,col=group.y;DSC sum={{0,0},{0,0}};
 for(ulong k=lane;k<p.npl;k+=lanes) {
   DSC a=load_dsc(wave_a,k+ulong(row)*p.npl);a.imag=ds_neg(a.imag);
   sum=dsc_add(sum,dsc_mul(a,load_dsc(wave_b,k+ulong(col)*p.npl)));
 }
 for(ulong k=lane;k<p.npro;k+=lanes) {
   DSC a=load_dsc(proj_a,k+ulong(row)*p.npro);a.imag=ds_neg(a.imag);
   sum=dsc_add(sum,dsc_mul(a,load_dsc(proj_b,k+ulong(col)*p.npro)));
 }
 threadgroup float4 partial[256];partial[lane]=store_dsc(sum);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=lanes/2;stride;stride/=2) {
   if(lane<stride) {
     float4 a=partial[lane],b=partial[lane+stride];
     DSC lhs={{a.x,a.y},{a.z,a.w}},rhs={{b.x,b.y},{b.z,b.w}};
     partial[lane]=store_dsc(dsc_add(lhs,rhs));
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(!lane) {
   ulong at=ulong(row)+ulong(col)*p.rows;float4 v=partial[0];
   DSC value={{v.x,v.y},{v.z,v.w}};
   overlap[at]=store_dsc(dsc_add(load_dsc(overlap,at),value));
 }
}
struct FullOrthParams {uint npl,npro,bands;};
struct FullOrthMetricParams {uint npro,bands,nions,lmdim;};
// Keep this ABI identical to HamSiteHost in metal_backend.mm.  The unused
// type field is retained so Metal's stride matches the host's 20-byte record.
struct FullOrthSite {uint ion,type,base,count,l;};
kernel void ds_full_orth_metric(device const float4* metric [[buffer(0)]],device const FullOrthSite* site [[buffer(1)]],
 device const float4* proj [[buffer(2)]],device float4* metric_proj [[buffer(3)]],
 constant FullOrthMetricParams& p [[buffer(4)]],uint2 grid [[thread_position_in_grid]]) {
 if(grid.x>=p.npro || grid.y>=p.bands)return;uint row=grid.x,band=grid.y;ulong i=ulong(row)+ulong(band)*p.npro;FullOrthSite sr=site[row];
 DSC sum={{0,0},{0,0}};
 for(uint col=sr.base;col<sr.base+sr.count;++col) {FullOrthSite sc=site[col];
   if(sr.ion==sc.ion) {ulong at=sr.l+ulong(p.lmdim)*sc.l+ulong(sr.ion)*p.lmdim*p.lmdim;
     DSC m=load_dsc(metric,at); DSC x=load_dsc(proj,ulong(band)*p.npro+col);
     sum=dsc_add(sum,dsc_mul(m,x));}
 }
 metric_proj[i]=store_dsc(sum);
}
kernel void ds_full_orth_gram(device const float4* wave [[buffer(0)]],device const float4* proj [[buffer(1)]],
 device const float4* metric_proj [[buffer(2)]],device float4* gram [[buffer(3)]],constant FullOrthParams& p [[buffer(4)]],uint2 grid [[thread_position_in_grid]]) {
 if(grid.x>=p.bands || grid.y>=p.bands)return;uint row=grid.x,col=grid.y;ulong i=ulong(row)+ulong(col)*p.bands;bool lower=row>col;
 if(lower) {uint t=row;row=col;col=t;}
 DSC sum={{0,0},{0,0}};
 for(uint g=0;g<p.npl;++g) sum=dsc_add(sum,dsc_mul(dsc_conj(load_dsc(wave,g+ulong(row)*p.npl)),load_dsc(wave,g+ulong(col)*p.npl)));
 for(uint j=0;j<p.npro;++j) sum=dsc_add(sum,dsc_mul(dsc_conj(load_dsc(proj,j+ulong(row)*p.npro)),load_dsc(metric_proj,j+ulong(col)*p.npro)));
 if(row==col) sum.imag={0,0};else if(lower) sum=dsc_conj(sum);
 gram[i]=store_dsc(sum);
}
kernel void ds_full_orth_update(device const float4* wave [[buffer(0)]],device const float4* proj [[buffer(1)]],
 device const float4* inv [[buffer(2)]],device float4* wave_out [[buffer(3)]],device float4* proj_out [[buffer(4)]],
 device const uint* state [[buffer(5)]],constant FullOrthParams& p [[buffer(6)]],uint2 grid [[thread_position_in_grid]]) {
 ulong i=ulong(grid.x)+ulong(grid.y)*65536ul,span=ulong(p.npl)+p.npro;
 if(i>=span*p.bands || state[0])return;
 ulong row=i%span;uint col=uint(i/span);DSC sum={{0,0},{0,0}};
 for(uint k=0;k<=col;++k) {
  DSC v=row<p.npl ? load_dsc(wave,row+ulong(k)*p.npl):load_dsc(proj,row-p.npl+ulong(k)*p.npro);
  sum=dsc_add(sum,dsc_mul(v,dsc_conj(load_dsc(inv,col+ulong(k)*p.bands))));
 }
 if(row<p.npl) wave_out[row+ulong(col)*p.npl]=store_dsc(sum);else proj_out[row-p.npl+ulong(col)*p.npro]=store_dsc(sum);
}
)METAL";
