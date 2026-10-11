#pragma once
// A complete Jacobi solve in one threadgroup for small projected matrices.
// H uses dynamic threadgroup memory; Q stays resident in a device buffer.
constexpr char kSubspaceShader[] = R"METAL(
kernel void ds_jacobi_local(device const float4 *a [[buffer(0)]],
                            device float4 *q [[buffer(1)]],
                            device float4 *diagonal [[buffer(2)]],
                            device uint *state [[buffer(3)]],
                            constant uint &n [[buffer(4)]],
                            threadgroup float4 *h [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[64];
    threadgroup uint pivots[64];
    threadgroup JacobiParams rotation;
    threadgroup uint active;
    for(uint index=tid;index<ulong(n)*n;index+=64u) {
        h[index]=a[index];
        q[index]=float4(index%n==index/n ? 1.0f:0.0f,0.0f,0.0f,0.0f);
    }
    if(tid==0u) { state[0]=1u;state[1]=0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);
    for(uint iteration=0;iteration<16u*ulong(n)*n+32u;++iteration) {
        float largest=0.0f;uint pivot=0u;
        for(uint index=tid;index<ulong(n)*n;index+=64u) if(index%n<index/n) {
            float4 x=h[index];float mag=length(float2(x.x+x.y,x.z+x.w));
            if(mag>largest) {largest=mag;pivot=index;}
        }
        maxima[tid]=largest;pivots[tid]=pivot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0u) {
            uint selected=0u;float maximum=0.0f,scale=1.0f;
            for(uint t=0;t<64u;++t) if(maxima[t]>maximum) {maximum=maxima[t];selected=pivots[t];}
            for(uint i=0;i<n;++i) scale=max(scale,abs(h[i+ulong(i)*n].x));
            active=maximum>1e-13f*scale ? 1u:0u;
            if(active!=0u) {
                uint p=selected%n,r=selected/n;float4 x=h[selected];
                DSC hpq={{x.x,x.y},{x.z,x.w}};
                DS mag=ds_sqrt(dsc_mul(dsc_conj(hpq),hpq).real);
                DS app={h[p+ulong(p)*n].x,h[p+ulong(p)*n].y},arr={h[r+ulong(r)*n].x,h[r+ulong(r)*n].y};
                DS tau=ds_div(ds_add(arr,ds_neg(app)),ds_add(mag,mag));
                DS abst=tau.hi<0.0f ? ds_neg(tau):tau;
                DS t=ds_div((DS){tau.hi>=0.0f ? 1.0f:-1.0f,0.0f},ds_add(abst,ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(tau,tau)))));
                DS c=ds_div((DS){1.0f,0.0f},ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(t,t))));
                DS factor=ds_div(ds_mul(c,t),mag);
                rotation.n=n;rotation.p=p;rotation.r=r;
                rotation.c=float2(c.hi,c.lo);
                rotation.s=store_dsc((DSC){ds_mul(hpq.real,factor),ds_mul(hpq.imag,factor)});
                DS dp=ds_add(app,ds_neg(ds_mul(t,mag))),dr=ds_add(arr,ds_mul(t,mag));
                rotation.pp=float2(dp.hi,dp.lo);rotation.rr=float2(dr.hi,dr.lo);
            } else state[0]=0u;
            state[1]=iteration;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(active==0u) break;
        uint p=rotation.p,r=rotation.r;
        DSC s={{rotation.s.x,rotation.s.y},{rotation.s.z,rotation.s.w}};
        for(uint k=tid;k<n;k+=64u) {
            if(k!=p && k!=r) {
                float4 ap=h[k+ulong(p)*n],ar=h[k+ulong(r)*n];
                DSC hp={{ap.x,ap.y},{ap.z,ap.w}},hr={{ar.x,ar.y},{ar.z,ar.w}};
                DSC vp=dsc_add(dsc_scale_ds(hp,rotation.c),dsc_scale(dsc_mul(dsc_conj(s),hr),-1.0f));
                DSC vr=dsc_add(dsc_scale_ds(hr,rotation.c),dsc_mul(s,hp));
                h[k+ulong(p)*n]=store_dsc(vp);h[p+ulong(k)*n]=store_dsc(dsc_conj(vp));
                h[k+ulong(r)*n]=store_dsc(vr);h[r+ulong(k)*n]=store_dsc(dsc_conj(vr));
            }
            DSC qp=load_dsc(q,k+ulong(p)*n),qr=load_dsc(q,k+ulong(r)*n);
            q[k+ulong(p)*n]=store_dsc(dsc_add(dsc_scale_ds(qp,rotation.c),dsc_scale(dsc_mul(dsc_conj(s),qr),-1.0f)));
            q[k+ulong(r)*n]=store_dsc(dsc_add(dsc_scale_ds(qr,rotation.c),dsc_mul(s,qp)));
        }
        if(tid==0u) {
            h[p+ulong(p)*n]=float4(rotation.pp.x,rotation.pp.y,0.0f,0.0f);
            h[r+ulong(r)*n]=float4(rotation.rr.x,rotation.rr.y,0.0f,0.0f);
            h[p+ulong(r)*n]=h[r+ulong(p)*n]=float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);
    }
    for(uint i=tid;i<n;i+=64u) diagonal[i+ulong(i)*n]=h[i+ulong(i)*n];
}

// Column Cholesky: one column is computed per dispatch, all rows in parallel.
// The positive diagonal pivot is computed separately before each column update.
kernel void ds_chol_pivot(device float4 *l [[buffer(0)]],device uint *state [[buffer(1)]],
                          constant uint &n [[buffer(2)]],constant uint &col [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if(index!=0u || state[0]!=0u) return;
    DS sum=load_dsc(l,col+ulong(col)*n).real;
    for(uint k=0;k<col;++k) {DSC x=load_dsc(l,col+ulong(k)*n);sum=ds_add(sum,ds_neg(dsc_mul(x,dsc_conj(x)).real));}
    if(!isfinite(sum.hi) || sum.hi<=0.0f) {state[0]=col+1u;return;}
    DS value=ds_sqrt(sum);l[col+ulong(col)*n]=float4(value.hi,value.lo,0.0f,0.0f);
}
kernel void ds_chol_column(device float4 *l [[buffer(0)]],device const uint *state [[buffer(1)]],
                           constant uint &n [[buffer(2)]],constant uint &col [[buffer(3)]],
                           uint row [[thread_position_in_grid]]) {
    if(row<=col || row>=n || state[0]!=0u) return;
    DSC sum=load_dsc(l,row+ulong(col)*n);
    for(uint k=0;k<col;++k) sum=dsc_add(sum,dsc_scale(dsc_mul(load_dsc(l,row+ulong(k)*n),dsc_conj(load_dsc(l,col+ulong(k)*n))),-1.0f));
    DSC diagonal=load_dsc(l,col+ulong(col)*n);
    l[row+ulong(col)*n]=store_dsc((DSC){ds_div(sum.real,diagonal.real),ds_div(sum.imag,diagonal.real)});
}
// Parallel right-hand sides of the triangular inverse stay on GPU.
kernel void ds_lower_inverse(device const float4 *l [[buffer(0)]],device float4 *inv [[buffer(1)]],
                             device const uint *state [[buffer(2)]],constant uint &n [[buffer(3)]],
                             uint col [[thread_position_in_grid]]) {
    if(col>=n || state[0]!=0u) return;
    for(uint row=0;row<n;++row) {
        DSC sum={{row==col ? 1.0f:0.0f,0.0f},{0.0f,0.0f}};
        for(uint k=0;k<row;++k) sum=dsc_add(sum,dsc_scale(dsc_mul(load_dsc(l,row+ulong(k)*n),load_dsc(inv,k+ulong(col)*n)),-1.0f));
        DSC diag=load_dsc(l,row+ulong(row)*n);
        inv[row+ulong(col)*n]=store_dsc((DSC){ds_div(sum.real,diag.real),ds_div(sum.imag,diag.real)});
    }
}
// Column-major product variants: 0=A B, 1=A B^H, 2=A^H B.
kernel void ds_subspace_product(device const float4 *a [[buffer(0)]],device const float4 *b [[buffer(1)]],
                                device float4 *c [[buffer(2)]],device const uint *state [[buffer(3)]],
                                constant uint &n [[buffer(4)]],constant uint &mode [[buffer(5)]],
                                uint2 grid [[thread_position_in_grid]]) {
    const ulong index=ulong(grid.x)+ulong(grid.y)*n;
    if(grid.x>=n || grid.y>=n || state[0]!=0u) return;
    uint row=index%n,col=index/n;DSC sum={{0.0f,0.0f},{0.0f,0.0f}};
    for(uint k=0;k<n;++k) {
        DSC left=mode==2u ? dsc_conj(load_dsc(a,k+ulong(row)*n)):load_dsc(a,row+ulong(k)*n);
        DSC right=mode==1u ? dsc_conj(load_dsc(b,col+ulong(k)*n)):load_dsc(b,k+ulong(col)*n);
        sum=dsc_add(sum,dsc_mul(left,right));
    }
    c[index]=store_dsc(sum);
}
kernel void ds_hermitianize(device const float4 *a [[buffer(0)]],device float4 *c [[buffer(1)]],
                            constant uint &n [[buffer(2)]],uint2 grid [[thread_position_in_grid]]) {
    const ulong index=ulong(grid.x)+ulong(grid.y)*n;
    if(grid.x>=n || grid.y>=n) return;
    uint row=index%n,col=index/n;
    DSC sum=dsc_scale(dsc_add(load_dsc(a,index),dsc_conj(load_dsc(a,col+ulong(row)*n))),0.5f);
    if(row==col) sum.imag={0.0f,0.0f};c[index]=store_dsc(sum);
}
)METAL";

constexpr char kCyclicShader[] = R"METAL(
struct CyclicRound { uint n,padded,round; };
inline uint cyclic_pair(uint slot,constant CyclicRound &p) {
    if(slot==p.padded-1u) return 0u;
    uint offset=(slot+p.padded-1u-p.round)%(p.padded-1u);
    return min(offset,p.padded-1u-offset);
}
kernel void ds_cyclic_params(device const float4 *h [[buffer(0)]],device JacobiParams *params [[buffer(1)]],
                             constant CyclicRound &p [[buffer(2)]],uint pair [[thread_position_in_grid]]) {
    if(pair>=p.padded/2u) return;
    uint u=(p.round+pair)%(p.padded-1u),v=pair==0u ? p.padded-1u:(p.round+p.padded-1u-pair)%(p.padded-1u);
    uint pp=min(u,v),rr=max(u,v);JacobiParams rot;rot.n=p.n;rot.p=pp;rot.r=rr;rot.reserved=0;
    rot.c=float2(1.0f,0.0f);rot.s=float4(0.0f);
    if(rr<p.n) {
        DSC hpq=load_dsc(h,pp+ulong(rr)*p.n);DS mag=ds_sqrt(dsc_mul(dsc_conj(hpq),hpq).real);
        DS app=load_dsc(h,pp+ulong(pp)*p.n).real,arr=load_dsc(h,rr+ulong(rr)*p.n).real;
        float scale=max(1.0f,max(abs(app.hi),abs(arr.hi)));
        if(mag.hi>1e-14f*scale) {
            DS tau=ds_div(ds_add(arr,ds_neg(app)),ds_add(mag,mag));DS abst=tau.hi<0.0f ? ds_neg(tau):tau;
            DS t=ds_div((DS){tau.hi>=0.0f ? 1.0f:-1.0f,0.0f},ds_add(abst,ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(tau,tau)))));
            DS c=ds_div((DS){1.0f,0.0f},ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(t,t))));DS f=ds_div(ds_mul(c,t),mag);
            rot.c=float2(c.hi,c.lo);rot.s=store_dsc((DSC){ds_mul(hpq.real,f),ds_mul(hpq.imag,f)});rot.reserved=1u;
        }
    }
    params[pair]=rot;
}
kernel void ds_cyclic_right(device const float4 *h [[buffer(0)]],device float4 *out [[buffer(1)]],
                            device const JacobiParams *params [[buffer(2)]],constant CyclicRound &p [[buffer(3)]],
                            uint2 grid [[thread_position_in_grid]]) {
    const ulong index=ulong(grid.x)+ulong(grid.y)*p.n;
    if(grid.x>=p.n || grid.y>=p.n) return;
    uint row=index%p.n,col=index/p.n;JacobiParams rot=params[cyclic_pair(col,p)];
    DSC value=load_dsc(h,index);
    if(rot.reserved!=0u) {
        DSC a=load_dsc(h,row+ulong(rot.p)*p.n),b=load_dsc(h,row+ulong(rot.r)*p.n),s={{rot.s.x,rot.s.y},{rot.s.z,rot.s.w}};
        value=col==rot.p ? dsc_add(dsc_scale_ds(a,rot.c),dsc_scale(dsc_mul(dsc_conj(s),b),-1.0f)) : dsc_add(dsc_scale_ds(b,rot.c),dsc_mul(s,a));
    }
    out[index]=store_dsc(value);
}
kernel void ds_cyclic_left(device const float4 *h [[buffer(0)]],device float4 *out [[buffer(1)]],
                           device const JacobiParams *params [[buffer(2)]],constant CyclicRound &p [[buffer(3)]],
                           uint2 grid [[thread_position_in_grid]]) {
    const ulong index=ulong(grid.x)+ulong(grid.y)*p.n;
    if(grid.x>=p.n || grid.y>=p.n) return;
    uint row=index%p.n,col=index/p.n;JacobiParams rot=params[cyclic_pair(row,p)];DSC value=load_dsc(h,index);
    if(rot.reserved!=0u) {
        DSC a=load_dsc(h,rot.p+ulong(col)*p.n),b=load_dsc(h,rot.r+ulong(col)*p.n),s={{rot.s.x,rot.s.y},{rot.s.z,rot.s.w}};
        value=row==rot.p ? dsc_add(dsc_scale_ds(a,rot.c),dsc_scale(dsc_mul(s,b),-1.0f)) : dsc_add(dsc_scale_ds(b,rot.c),dsc_mul(dsc_conj(s),a));
    }
    if(row==col) value.imag={0.0f,0.0f};out[index]=store_dsc(value);
}
kernel void ds_cyclic_check(device const float4 *h [[buffer(0)]],device uint *state [[buffer(1)]],
                            constant uint &n [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    if(index!=0u) return;
    float largest=0.0f,scale=1.0f;
    for(uint j=0;j<n;++j) {
        scale=max(scale,abs(h[j+ulong(j)*n].x));
        for(uint i=0;i<j;++i) {float4 x=h[i+ulong(j)*n];largest=max(largest,length(float2(x.x+x.y,x.z+x.w)));}
    }
    state[0]=isfinite(largest) && largest<2e-13f*scale ? 0u:1u;
}
)METAL";
