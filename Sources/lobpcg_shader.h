#pragma once
// Genuine block LOBPCG on a dense projected Hermitian operator (block size 2).
// All iteration vectors, projected Ritz matrices and deflation stay on Metal.
// It is not a replacement for VASP's full plane-wave Hamiltonian or PAW metric.
constexpr char kLobpcgShader[] = R"METAL(
struct LobState { uint n,locked,k,dim,status,iterations; float scale,residual; };
inline DSC lob_zero() { return {{0.0f,0.0f},{0.0f,0.0f}}; }
inline DSC lob_sub(DSC a,DSC b) { return dsc_add(a,dsc_scale(b,-1.0f)); }
inline DS lob_norm(device const float4 *s,uint n,uint col) {
    DS sum={0.0f,0.0f};
    for(uint i=0;i<n;++i) { DSC x=load_dsc(s,i+ulong(col)*n); sum=ds_add(sum,dsc_mul(dsc_conj(x),x).real); }
    return ds_sqrt(sum);
}
inline void lob_orthogonalize(device float4 *s,device const float4 *locked,device LobState *st,uint count) {
    uint n=st->n,kept=0;
    for(uint col=0;col<count;++col) {
        for(uint pass=0;pass<2;++pass) {
            for(uint j=0;j<st->locked+kept;++j) {
                DSC dot=lob_zero();
                for(uint i=0;i<n;++i) {
                    DSC v=j<st->locked ? load_dsc(locked,i+ulong(j)*n) : load_dsc(s,i+ulong(j-st->locked)*n);
                    dot=dsc_add(dot,dsc_mul(dsc_conj(v),load_dsc(s,i+ulong(col)*n)));
                }
                for(uint i=0;i<n;++i) {
                    DSC v=j<st->locked ? load_dsc(locked,i+ulong(j)*n) : load_dsc(s,i+ulong(j-st->locked)*n);
                    s[i+ulong(col)*n]=store_dsc(lob_sub(load_dsc(s,i+ulong(col)*n),dsc_mul(v,dot)));
                }
            }
        }
        DS norm=lob_norm(s,n,col);
        if(norm.hi>1e-13f) {
            DS inv=ds_div((DS){1.0f,0.0f},norm);
            for(uint i=0;i<n;++i) s[i+ulong(kept)*n]=store_dsc(dsc_scale_ds(load_dsc(s,i+ulong(col)*n),float2(inv.hi,inv.lo)));
            ++kept;
        }
    }
    st->dim=kept;
    if(kept<st->k) st->status=2;
}
inline void lob_initialize(device float4 *s,device const float4 *locked,device LobState *st) {
    st->k=min(2u,st->n-st->locked);
    for(uint col=0;col<st->k;++col) for(uint i=0;i<st->n;++i) {
        // Deterministic nontrivial starts. FP32 here only sets initial guesses.
        float t=float((i+1u)*(st->locked+col+3u));
        s[i+col*st->n]=float4(sin(t*1.731f),0.0f,cos(t*0.937f),0.0f);
    }
    lob_orthogonalize(s,locked,st,st->k);
}
// Small projected Ritz solve, independent of host LAPACK. Complex DS Jacobi.
inline void lob_ritz_solve(thread DSC *h,thread DSC *q,uint d) {
    for(uint j=0;j<d;++j) for(uint i=0;i<d;++i) q[i+j*d]={{i==j ? 1.0f:0.0f,0.0f},{0.0f,0.0f}};
    for(uint it=0;it<256u;++it) {
        uint p=0,r=0;float largest=0.0f,scale=1.0f;
        for(uint j=0;j<d;++j) {
            scale=max(scale,abs(h[j+j*d].real.hi));
            for(uint i=0;i<j;++i) {
                DSC x=h[i+j*d]; float mag=length(float2(x.real.hi,x.imag.hi));
                if(mag>largest) {largest=mag;p=i;r=j;}
            }
        }
        if(largest<1e-14f*scale) break;
        DSC hpq=h[p+r*d];DS mag=ds_sqrt(dsc_mul(dsc_conj(hpq),hpq).real);
        DS app=h[p+p*d].real,arr=h[r+r*d].real;
        DS tau=ds_div(ds_add(arr,ds_neg(app)),ds_add(mag,mag));
        DS abst=tau.hi<0.0f ? ds_neg(tau):tau;
        DS t=ds_div((DS){tau.hi>=0.0f ? 1.0f:-1.0f,0.0f},ds_add(abst,ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(tau,tau)))));
        DS c=ds_div((DS){1.0f,0.0f},ds_sqrt(ds_add((DS){1.0f,0.0f},ds_mul(t,t))));
        DS factor=ds_div(ds_mul(c,t),mag);DSC ss={ds_mul(hpq.real,factor),ds_mul(hpq.imag,factor)};
        for(uint k=0;k<d;++k) if(k!=p && k!=r) {
            DSC a=h[k+p*d],b=h[k+r*d];
            h[k+p*d]=lob_sub(dsc_scale_ds(a,float2(c.hi,c.lo)),dsc_mul(dsc_conj(ss),b));
            h[p+k*d]=dsc_conj(h[k+p*d]);
            h[k+r*d]=dsc_add(dsc_scale_ds(b,float2(c.hi,c.lo)),dsc_mul(ss,a));
            h[r+k*d]=dsc_conj(h[k+r*d]);
        }
        h[p+p*d]={ds_add(app,ds_neg(ds_mul(t,mag))),{0.0f,0.0f}};
        h[r+r*d]={ds_add(arr,ds_mul(t,mag)),{0.0f,0.0f}};
        h[p+r*d]=h[r+p*d]=lob_zero();
        for(uint k=0;k<d;++k) {
            DSC a=q[k+p*d],b=q[k+r*d];
            q[k+p*d]=lob_sub(dsc_scale_ds(a,float2(c.hi,c.lo)),dsc_mul(dsc_conj(ss),b));
            q[k+r*d]=dsc_add(dsc_scale_ds(b,float2(c.hi,c.lo)),dsc_mul(ss,a));
        }
    }
}
kernel void ds_lob_init(device float4 *s [[buffer(0)]],device const float4 *locked [[buffer(1)]],device LobState *st [[buffer(2)]],device const float4 *a [[buffer(3)]],uint index [[thread_position_in_grid]]) {
    if(index==0u) {
        float scale=1.0f;
        for(uint i=0;i<st->n*st->n;++i) scale=max(scale,length(float2(a[i].x,a[i].z)));
        st->scale=scale*float(st->n);
        lob_initialize(s,locked,st);
    }
}
kernel void ds_lob_apply(device const float4 *a [[buffer(0)]],device const float4 *s [[buffer(1)]],device float4 *as [[buffer(2)]],device const LobState *st [[buffer(3)]],uint index [[thread_position_in_grid]]) {
    if(st->status!=1u || index>=st->n*st->dim) return;
    uint row=index%st->n,col=index/st->n;DSC sum=lob_zero();
    for(uint j=0;j<st->n;++j) sum=dsc_add(sum,dsc_mul(load_dsc(a,row+j*st->n),load_dsc(s,j+col*st->n)));
    as[index]=store_dsc(sum);
}
kernel void ds_lob_update(device const float4 *a [[buffer(0)]],device float4 *s [[buffer(1)]],device const float4 *as [[buffer(2)]],device float4 *locked [[buffer(3)]],device float2 *values [[buffer(4)]],device float4 *tmp [[buffer(5)]],device LobState *st [[buffer(6)]],uint index [[thread_position_in_grid]]) {
    if(index!=0u || st->status!=1u) return;
    uint n=st->n,d=st->dim,k=st->k;DSC h[36],q[36];
    for(uint j=0;j<d;++j) for(uint i=0;i<=j;++i) {
        DSC dot=lob_zero();
        for(uint r=0;r<n;++r) dot=dsc_add(dot,dsc_mul(dsc_conj(load_dsc(s,r+ulong(i)*n)),load_dsc(as,r+ulong(j)*n)));
        if(i==j) dot.imag={0.0f,0.0f};
        h[i+j*d]=dot;h[j+i*d]=dsc_conj(dot);
    }
    lob_ritz_solve(h,q,d);uint order[6];
    for(uint i=0;i<d;++i) order[i]=i;
    for(uint i=0;i<d;++i) for(uint j=i+1;j<d;++j)
        if(h[order[j]+order[j]*d].real.hi<h[order[i]+order[i]*d].real.hi) {uint old=order[i];order[i]=order[j];order[j]=old;}
    float largest=0.0f;
    for(uint c=0;c<k;++c) {
        DS lambda=h[order[c]+order[c]*d].real;
        DS norm={0.0f,0.0f};
        for(uint r=0;r<n;++r) {
            DSC x=lob_zero(),ax=lob_zero(),p=lob_zero();
            for(uint j=0;j<d;++j) {
                DSC coeff=q[j+order[c]*d];
                x=dsc_add(x,dsc_mul(load_dsc(s,r+ulong(j)*n),coeff));
                ax=dsc_add(ax,dsc_mul(load_dsc(as,r+ulong(j)*n),coeff));
                if(j>=k) p=dsc_add(p,dsc_mul(load_dsc(s,r+ulong(j)*n),coeff));
            }
            DSC residual=lob_sub(ax,dsc_scale_ds(x,float2(lambda.hi,lambda.lo)));
            tmp[r+ulong(c)*n]=store_dsc(x);
            tmp[r+ulong(k+c)*n]=store_dsc(residual);
            tmp[r+ulong(2*k+c)*n]=store_dsc(p);
        }
        // Deflation solves the operator restricted to the unlocked subspace.
        // Tiny residuals of locked Ritz vectors must not prevent later blocks
        // from converging. Final full-space backward error is checked on host.
        for(uint j=0;j<st->locked;++j) {
            DSC dot=lob_zero();
            for(uint r=0;r<n;++r) dot=dsc_add(dot,dsc_mul(dsc_conj(load_dsc(locked,r+ulong(j)*n)),load_dsc(tmp,r+ulong(k+c)*n)));
            for(uint r=0;r<n;++r) tmp[r+ulong(k+c)*n]=store_dsc(lob_sub(load_dsc(tmp,r+ulong(k+c)*n),dsc_mul(load_dsc(locked,r+ulong(j)*n),dot)));
        }
        for(uint r=0;r<n;++r) {
            DSC residual=load_dsc(tmp,r+ulong(k+c)*n);
            norm=ds_add(norm,dsc_mul(dsc_conj(residual),residual).real);
            DS diag=load_dsc(a,r+ulong(r)*n).real;if(diag.hi<0.0f) diag=ds_neg(diag);
            DS inv=ds_div((DS){-1.0f,0.0f},ds_add(diag,(DS){st->scale/float(n),0.0f}));
            tmp[r+ulong(k+c)*n]=store_dsc(dsc_scale_ds(residual,float2(inv.hi,inv.lo)));
        }
        largest=max(largest,ds_sqrt(norm).hi);
        values[st->locked+c]=float2(lambda.hi,lambda.lo);
    }
    ++st->iterations;st->residual=largest;
    if(!isfinite(largest)) {st->status=2;return;}
    if(largest<=1e-13f*st->scale) {
        for(uint c=0;c<k;++c) for(uint r=0;r<n;++r) locked[r+ulong(st->locked+c)*n]=tmp[r+ulong(c)*n];
        st->locked+=k;
        if(st->locked==n) {st->status=0;return;}
        lob_initialize(s,locked,st);
    } else {
        for(uint c=0;c<3*k;++c) for(uint r=0;r<n;++r) s[r+ulong(c)*n]=tmp[r+ulong(c)*n];
        lob_orthogonalize(s,locked,st,3*k);
    }
}
kernel void ds_lob_diagonal(device const float2 *values [[buffer(0)]],device float4 *diagonal [[buffer(1)]],constant uint &n [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    if(index<n) {float2 x=values[index];diagonal[index+ulong(index)*n]=float4(x.x,x.y,0.0f,0.0f);}
}
)METAL";
