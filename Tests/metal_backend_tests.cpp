#include "metal_backend.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <limits>
#include <vector>
#include <string>

static metal_complex64 z(double r, double i) { return {r, i}; }

static void test_resident(metal_backend *backend) {
    using cd=std::complex<double>;
    constexpr size_t total=16,bands=2,ld=19,npro=2;
    metal_resident *r=nullptr;
    assert(metal_backend_resident_create(backend,4,2,2,bands,npro,&r)==0);
    std::vector<metal_complex64> wave(ld*bands,z(123,-456)),out(wave),potential(total),projectors(ld*npro),coeff(3*bands,z(123,-456));
    for(size_t b=0;b<bands;++b) for(size_t g=0;g<total;++g) wave[g+b*ld]=z(std::sin(.17*(g+3*b)),std::cos(.13*(g+7*b)));
    for(size_t g=0;g<total;++g) potential[g]=z(1.+.02*g,.003*g);
    for(size_t p=0;p<npro;++p) for(size_t g=0;g<total;++g) projectors[g+p*ld]=z(.03*(g+p+1),-.02*(g+2*p));
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==METAL_BACKEND_INVALID_ARGUMENT);
    const auto commands=metal_backend_command_count(backend);
    assert(metal_backend_resident_begin_step(r)==0);
    assert(metal_backend_resident_begin_step(r)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_resident_upload_wavefunction(r,wave.data(),ld,0)==0);
    assert(metal_backend_resident_apply_hamiltonian(r)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_resident_project(r)==METAL_BACKEND_INVALID_ARGUMENT);
    // Independent validity: first force=0 upload must initialize each input.
    assert(metal_backend_resident_upload_potential(r,potential.data(),0)==0);
    assert(metal_backend_resident_upload_projectors(r,projectors.data(),ld,0)==0);
    assert(metal_backend_resident_uploads(r)==3);
    assert(metal_backend_resident_fft(r,-1)==0);
    assert(metal_backend_resident_upload_wavefunction(r,wave.data(),ld,1)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_resident_upload_potential(r,potential.data(),1)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(out[0].real==123 && out[0].imag==-456);
    assert(metal_backend_resident_end_step(r,0)==0);
    // This shape has an odd number of FFT stages. Keeping the last ping-pong
    // result across begin_step is required to recover the original columns.
    assert(metal_backend_resident_begin_step(r)==0);
    assert(metal_backend_resident_upload_wavefunction(r,wave.data(),ld,0)==0);
    assert(metal_backend_resident_fft(r,1)==0);
    assert(metal_backend_resident_end_step(r,1)==0);
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==0);
    for(size_t b=0;b<bands;++b) for(size_t g=0;g<ld;++g) {
        const auto expected=g<total ? cd(wave[g+b*ld].real,wave[g+b*ld].imag)*double(total):cd(123,-456);
        assert(std::abs(cd(out[g+b*ld].real,out[g+b*ld].imag)-expected)<2e-9);
    }
    assert(metal_backend_command_count(backend)==commands+2);
    assert(metal_backend_resident_uploads(r)==3);
    assert(metal_backend_resident_steps(r)==2 && metal_backend_resident_submissions(r)==2);
    // Local potential + Euclidean orthogonalization + dense projector dot
    // in one submission, against an independent CPU complex reference.
    std::vector<cd> reference(total*bands);
    for(size_t b=0;b<bands;++b) for(size_t g=0;g<total;++g)
        reference[g+b*total]=cd(wave[g+b*ld].real,wave[g+b*ld].imag)*cd(potential[g].real,potential[g].imag);
    for(size_t b=0;b<bands;++b) {
        for(int pass=0;pass<2;++pass) for(size_t p=0;p<b;++p) {
            cd dot=0;for(size_t g=0;g<total;++g) dot+=std::conj(reference[g+p*total])*reference[g+b*total];
            for(size_t g=0;g<total;++g) reference[g+b*total]-=reference[g+p*total]*dot;
        }
        double norm=0;for(size_t g=0;g<total;++g) norm+=std::norm(reference[g+b*total]);
        for(size_t g=0;g<total;++g) reference[g+b*total]/=std::sqrt(norm);
    }
    assert(metal_backend_resident_begin_step(r)==0);
    assert(metal_backend_resident_upload_wavefunction(r,wave.data(),ld,1)==0);
    assert(metal_backend_resident_apply_hamiltonian(r)==0);
    assert(metal_backend_resident_orthogonalize(r)==0);
    assert(metal_backend_resident_project(r)==0);
    assert(metal_backend_resident_end_step(r,0)==0);
    assert(metal_backend_resident_download_projection(r,coeff.data(),3)==0);
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==0);
    for(size_t b=0;b<bands;++b) {
        for(size_t g=0;g<total;++g) assert(std::abs(cd(out[g+b*ld].real,out[g+b*ld].imag)-reference[g+b*total])<3e-10);
        for(size_t p=0;p<npro;++p) {
            cd expected=0;for(size_t g=0;g<total;++g) expected+=std::conj(cd(projectors[g+p*ld].real,projectors[g+p*ld].imag))*reference[g+b*total];
            assert(std::abs(cd(coeff[p+b*3].real,coeff[p+b*3].imag)-expected)<3e-10);
        }
        assert(coeff[2+b*3].real==123 && coeff[2+b*3].imag==-456);
    }
    assert(metal_backend_command_count(backend)==commands+3);
    // Aborting must discard unsubmitted output and permit a fresh upload.
    assert(metal_backend_resident_begin_step(r)==0);
    assert(metal_backend_resident_fft(r,-1)==0);
    assert(metal_backend_resident_abort_step(r)==0);
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_command_count(backend)==commands+3);
    assert(metal_backend_resident_begin_step(r)==0);
    std::fill(wave.begin(),wave.end(),z(0,0));
    assert(metal_backend_resident_upload_wavefunction(r,wave.data(),ld,0)==0);
    assert(metal_backend_resident_orthogonalize(r)==0);
    assert(metal_backend_resident_end_step(r,1)==METAL_BACKEND_RUNTIME_FAILED);
    assert(metal_backend_resident_download_wavefunction(r,out.data(),ld)==METAL_BACKEND_INVALID_ARGUMENT);
    assert(metal_backend_resident_begin_step(r)==0);
    assert(metal_backend_resident_abort_step(r)==0);
    metal_backend_resident_destroy(r);
    // Open-step destruction must never submit work or unlock a freed mutex.
    assert(metal_backend_resident_create(backend,2,1,1,1,0,&r)==0);
    assert(metal_backend_resident_begin_step(r)==0);
    metal_backend_resident_destroy(r);
    std::puts("resident async lifecycle, cached inputs, FFT round trip, projection and orthogonality passed");
}

int main() {
    metal_backend *backend = nullptr;
    const int create_code = metal_backend_create(&backend, 1);
    std::printf("create_code=%d backend_ptr_after_create=%p\n", create_code, static_cast<void *>(backend));
    assert(create_code == METAL_BACKEND_OK);
    assert(backend != nullptr);
    std::printf("backend_ptr=%p gpu_enabled=%d\n", static_cast<void *>(backend), metal_backend_gpu_enabled(backend));
    if (std::getenv("METAL_REQUIRE_GPU")) assert(metal_backend_gpu_enabled(backend));
    if (!metal_backend_gpu_enabled(backend))
        std::printf("gpu_fallback_reason=%s\n", metal_backend_last_error(backend));

    metal_resident *invalid=reinterpret_cast<metal_resident *>(uintptr_t(1));
    assert(metal_backend_resident_create(backend,0,1,1,1,0,&invalid)==METAL_BACKEND_INVALID_ARGUMENT && invalid==nullptr);
    assert(metal_backend_resident_create(backend,1,1,1,1,0,nullptr)==METAL_BACKEND_INVALID_ARGUMENT);
    if (metal_backend_gpu_enabled(backend)) test_resident(backend);
    else {
        assert(metal_backend_resident_create(backend,2,1,1,1,0,&invalid)==METAL_BACKEND_UNSUPPORTED && invalid==nullptr);
        std::puts("SKIP resident kernels: no Metal device visible; invalid arguments and unavailable handle verified");
    }

    // Column-major 2x2 complex GEMM against a direct reference.
    const metal_complex64 a[] = {z(1, 2), z(3, -1), z(-2, 0.5), z(4, 1)};
    const metal_complex64 b[] = {z(0.5, -1), z(2, 3), z(-1, 2), z(1.5, -0.25)};
    metal_complex64 c[4]{};
    assert(metal_backend_zgemm(backend, 2, 2, 2, a, 2, b, 2, c, 2) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t col = 0; col < 2; ++col) for (size_t row = 0; row < 2; ++row) {
        std::complex<double> expected{0, 0};
        for (size_t inner = 0; inner < 2; ++inner) {
            std::complex<double> av{a[row + inner * 2].real, a[row + inner * 2].imag};
            std::complex<double> bv{b[inner + col * 2].real, b[inner + col * 2].imag};
            expected += av * bv;
        }
        const auto got = std::complex<double>{c[row + col * 2].real, c[row + col * 2].imag};
        assert(std::abs(got - expected) < 1e-10);
    }
    // beta=0 cannot read a NaN/uninitialized C workspace (real VASP behavior).
    for (auto &x : c) x = z(std::numeric_limits<double>::quiet_NaN(), 1e200);
    assert(metal_backend_zgemm(backend, 2, 2, 2, a, 2, b, 2, c, 2) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (auto x : c) assert(std::isfinite(x.real) && std::isfinite(x.imag));

    // Exercise full BLAS semantics: transpose, conjugate-transpose, alpha, beta, and padding.
    const metal_complex64 ap[] = {z(1, 2), z(3, -1), z(9, 9), z(-2, 0.5), z(4, 1), z(8, 8)};
    const metal_complex64 bp[] = {z(0.5, -1), z(2, 3), z(7, 7), z(-1, 2), z(1.5, -0.25), z(6, 6)};
    metal_complex64 cp[4] = {z(0.25, -0.5), z(-0.75, 0.25), z(1.0, 0.5), z(-0.25, 0.75)};
    const metal_complex64 alpha{0.75, -0.25};
    const metal_complex64 beta{0.5, 0.125};
    assert(metal_backend_zgemm_blas(backend, 'C', 'N', 2, 2, 2, alpha, ap, 3, bp, 3, beta, cp, 2) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t col = 0; col < 2; ++col) for (size_t row = 0; row < 2; ++row) {
        std::complex<double> expected{0, 0};
        for (size_t inner = 0; inner < 2; ++inner) {
            const auto av = std::conj(std::complex<double>{ap[inner + row * 3].real, ap[inner + row * 3].imag});
            const auto bv = std::complex<double>{bp[inner + col * 3].real, bp[inner + col * 3].imag};
            expected += av * bv;
        }
        const auto old = std::complex<double>{(row + col * 2 == 0 ? 0.25 : row + col * 2 == 1 ? -0.75 : row + col * 2 == 2 ? 1.0 : -0.25),
                                              (row + col * 2 == 0 ? -0.5 : row + col * 2 == 1 ? 0.25 : row + col * 2 == 2 ? 0.5 : 0.75)};
        expected = std::complex<double>{alpha.real, alpha.imag} * expected + std::complex<double>{beta.real, beta.imag} * old;
        const auto got = std::complex<double>{cp[row + col * 2].real, cp[row + col * 2].imag};
        assert(std::abs(got - expected) < 1e-10);
    }

    std::vector<metal_complex64> input(16), transformed(16), recovered(16);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = z(std::sin(0.17 * static_cast<double>(i)), std::cos(0.23 * static_cast<double>(i)));
    assert(metal_backend_zfft(backend, input.size(), input.data(), transformed.data(), -1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    assert(metal_backend_zfft(backend, input.size(), transformed.data(), recovered.data(), +1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t i = 0; i < input.size(); ++i) {
        assert(std::abs(recovered[i].real - input[i].real) < 2e-5);
        assert(std::abs(recovered[i].imag - input[i].imag) < 2e-5);
    }
    // Compare the forward result against an independent Float64 direct DFT.
    for (size_t q = 0; q < input.size(); ++q) {
        std::complex<double> expected{0.0, 0.0};
        for (size_t p = 0; p < input.size(); ++p) {
            const std::complex<double> value{input[p].real, input[p].imag};
            const double angle = -2.0 * std::acos(-1.0) * static_cast<double>(p * q) / static_cast<double>(input.size());
            expected += value * std::complex<double>{std::cos(angle), std::sin(angle)};
        }
        const std::complex<double> got{transformed[q].real, transformed[q].imag};
        assert(std::abs(got - expected) < 2e-4);
    }
    // 3D contiguous layout round-trip and direct-reference check.
    const size_t nx = 4, ny = 2, nz = 2, total3 = nx * ny * nz;
    std::vector<metal_complex64> input3(total3), forward3(total3), recovered3(total3);
    for (size_t i = 0; i < total3; ++i) input3[i] = z(std::sin(0.11 * i), std::cos(0.07 * i));
    assert(metal_backend_zfft3d(backend, nx, ny, nz, input3.data(), forward3.data(), -1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    assert(metal_backend_zfft3d(backend, nx, ny, nz, forward3.data(), recovered3.data(), +1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t i = 0; i < total3; ++i) {
        assert(std::abs(recovered3[i].real - input3[i].real) < 3e-10);
        assert(std::abs(recovered3[i].imag - input3[i].imag) < 3e-10);
    }
    // Batched VASP/FFTW layout: transforms are separated by LDC and padding
    // must survive the call. This exercises the ABI used by FFTBAS_MU.
    const size_t batch_count = 3, batch_stride = total3 + 5;
    std::vector<metal_complex64> batch(batch_count * batch_stride);
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < total3; ++i)
            batch[b * batch_stride + i] = z(std::sin(0.07 * (i + 3 * b)), std::cos(0.13 * (i + b)));
        for (size_t i = total3; i < batch_stride; ++i) batch[b * batch_stride + i] = z(1234.0, -5678.0);
    }
    std::vector<metal_complex64> batch_reference(batch);
    for (size_t b = 0; b < batch_count; ++b) {
        std::vector<metal_complex64> one(total3);
        for (size_t i = 0; i < total3; ++i) one[i] = batch_reference[b * batch_stride + i];
        assert(metal_backend_zfft3d(backend, nx, ny, nz, one.data(), one.data(), -1) == METAL_BACKEND_OK);
        for (size_t i = 0; i < total3; ++i) batch_reference[b * batch_stride + i] = one[i];
    }
    assert(metal_backend_zfft3d_batch(backend, nx, ny, nz, batch_count, batch_stride,
                                      batch.data(), batch.data(), -1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < total3; ++i) {
            assert(std::abs(batch[b * batch_stride + i].real - batch_reference[b * batch_stride + i].real) < 3e-10);
            assert(std::abs(batch[b * batch_stride + i].imag - batch_reference[b * batch_stride + i].imag) < 3e-10);
        }
        for (size_t i = total3; i < batch_stride; ++i) {
            assert(batch[b * batch_stride + i].real == 1234.0);
            assert(batch[b * batch_stride + i].imag == -5678.0);
        }
    }
    // Odd batches expose byte-offset alignment and capacity reuse errors.
    for(size_t side: {3u,5u,1u,9u,3u}) {
        size_t len=side*side*side,stride=len+7,count=4;
        std::vector<metal_complex64> data(count*stride,z(1234,-5678)),saved;
        for(size_t b=0;b<count;++b) for(size_t i=0;i<len;++i) data[b*stride+i]=z(std::sin(0.1*(i+b)),std::cos(0.13*(i+3*b)));
        saved=data;
        uint64_t submits=metal_backend_command_count(backend);
        assert(metal_backend_zfft3d_batch(backend,side,side,side,count,stride,data.data(),data.data(),-1)==0);
        if(metal_backend_gpu_enabled(backend)) assert(metal_backend_command_count(backend)==submits+1);
        assert(metal_backend_zfft3d_batch(backend,side,side,side,count,stride,data.data(),data.data(),1)==0);
        for(size_t b=0;b<count;++b) for(size_t i=0;i<stride;++i) {
            assert(std::abs(data[b*stride+i].real-saved[b*stride+i].real)<2e-9);
            assert(std::abs(data[b*stride+i].imag-saved[b*stride+i].imag)<2e-9);
        }
        // Alternate a scalar transform with the next smaller batch.
        std::vector<metal_complex64> one(512,z(0.1,-0.2));
        assert(metal_backend_zfft3d(backend,8,8,8,one.data(),one.data(),-1)==0);
    }
    if(metal_backend_gpu_enabled(backend)) {
        std::vector<metal_complex64> rejected(17,z(0.2,-0.4));auto saved=rejected;
        assert(metal_backend_zfft3d_batch_gpu(backend,17,1,1,1,17,rejected.data(),rejected.data(),-1,0)==METAL_BACKEND_UNSUPPORTED);
        for(size_t i=0;i<17;++i) assert(rejected[i].real==saved[i].real && rejected[i].imag==saved[i].imag);
        assert(metal_backend_zfft3d_batch_gpu(backend,SIZE_MAX,2,2,2,17,rejected.data(),rejected.data(),-1,0)==METAL_BACKEND_INVALID_ARGUMENT);
    }
    // Non-power-of-two sizes use the correctness DFT fallback.
    const size_t non_power_two_n = 6;
    std::vector<metal_complex64> np_input(non_power_two_n), np_forward(non_power_two_n), np_recovered(non_power_two_n);
    for (size_t i = 0; i < non_power_two_n; ++i) np_input[i] = z(0.2 * static_cast<double>(i), -0.1 * static_cast<double>(i + 1));
    assert(metal_backend_zfft(backend, non_power_two_n, np_input.data(), np_forward.data(), -1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    assert(metal_backend_zfft(backend, non_power_two_n, np_forward.data(), np_recovered.data(), +1) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    for (size_t i = 0; i < non_power_two_n; ++i) {
        assert(std::abs(np_recovered[i].real - np_input[i].real) < 2e-10);
        assert(std::abs(np_recovered[i].imag - np_input[i].imag) < 2e-10);
    }

    // Combined plane-wave/PAW metric overlap, padded Fortran leading dimensions.
    {
      const size_t rows=5,cols=3,np=11,npro=4,ldw=14,ldp=7,ldo=8;
      std::vector<metal_complex64> wa(ldw*rows),wb(ldw*cols),pa(ldp*rows),pb(ldp*cols),over(ldo*cols,z(123,-456)),reference;
      for(size_t i=0;i<wa.size();++i) wa[i]=z(std::sin(0.1*i),std::cos(0.13*i));
      for(size_t i=0;i<wb.size();++i) wb[i]=z(std::sin(0.09*i),std::cos(0.17*i));
      for(size_t i=0;i<pa.size();++i) pa[i]=z(0.02*i,-0.03*i);
      for(size_t i=0;i<pb.size();++i) pb[i]=z(-0.05*i,0.04*i);
      for(size_t j=0;j<cols;++j) for(size_t i=0;i<rows;++i) over[i+j*ldo]=z(0.2*i,-0.1*j);
      reference=over;
      for(size_t j=0;j<cols;++j) for(size_t i=0;i<rows;++i){
        std::complex<double> sum{over[i+j*ldo].real,over[i+j*ldo].imag};
        for(size_t k=0;k<np;++k) sum+=std::conj(std::complex<double>{wa[k+i*ldw].real,wa[k+i*ldw].imag})*std::complex<double>{wb[k+j*ldw].real,wb[k+j*ldw].imag};
        for(size_t k=0;k<npro;++k) sum+=std::conj(std::complex<double>{pa[k+i*ldp].real,pa[k+i*ldp].imag})*std::complex<double>{pb[k+j*ldp].real,pb[k+j*ldp].imag};
        reference[i+j*ldo]=z(sum.real(),sum.imag());
      }
      if(metal_backend_gpu_enabled(backend)) {
        auto submits=metal_backend_command_count(backend);
        assert(metal_backend_orth_overlap_gpu(backend,rows,cols,np,npro,wa.data(),wb.data(),ldw,pa.data(),pb.data(),ldp,over.data(),ldo)==0);
        assert(metal_backend_command_count(backend)==submits+1);
        for(size_t i=0;i<over.size();++i) {assert(std::abs(over[i].real-reference[i].real)<1e-11);assert(std::abs(over[i].imag-reference[i].imag)<1e-11);}
      }
    }
    std::vector<double> values(10000);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (i % 3 == 0 ? 1.0 : (i % 3 == 1 ? -0.25 : 0.125));
    double sum = 0.0;
    assert(metal_backend_dsum(backend, values.size(), values.data(), &sum) == METAL_BACKEND_OK);
    if (metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
    double expected_sum = 0.0;
    for (double value : values) expected_sum += value;
    assert(std::abs(sum - expected_sum) < 1e-8);

    // Complex conjugation, non-unit/negative strides and cancellation must
    // agree with an independent long-double accumulator.
    for(size_t dot_n : {1u,255u,257u,65537u}) {
        std::vector<metal_complex64> xx(dot_n*2),yy(dot_n*3);
        for(size_t i=0;i<xx.size();++i) xx[i]=z(std::sin(.017*i),std::cos(.031*i));
        for(size_t i=0;i<yy.size();++i) yy[i]=z(std::cos(.009*i),std::sin(.023*i));
        for(auto increments : {std::pair<ptrdiff_t,ptrdiff_t>{1,1},{2,3},{-2,3},{2,-3}}) {
            auto ix=increments.first,iy=increments.second;
            ptrdiff_t sx=ix<0 ? (1-static_cast<ptrdiff_t>(dot_n))*ix:0;
            ptrdiff_t sy=iy<0 ? (1-static_cast<ptrdiff_t>(dot_n))*iy:0;
            std::complex<long double> reference=0.;
            for(size_t i=0;i<dot_n;++i) {
                auto a=xx[sx+static_cast<ptrdiff_t>(i)*ix],b=yy[sy+static_cast<ptrdiff_t>(i)*iy];
                reference+=std::conj(std::complex<long double>(a.real,a.imag))*std::complex<long double>(b.real,b.imag);
            }
            metal_complex64 dot{};
            assert(metal_backend_zdotc(backend,dot_n,xx.data(),ix,yy.data(),iy,&dot)==METAL_BACKEND_OK);
            if(metal_backend_gpu_enabled(backend)) assert(metal_backend_last_gpu_success(backend));
            double error=std::abs(std::complex<double>(dot.real,dot.imag)-std::complex<double>(reference.real(),reference.imag()));
            std::printf("zdotc n=%zu strides=%td,%td absolute_error=%.3e\n",dot_n,ix,iy,error);
            assert(error<2e-9);
        }
    }

    const metal_complex64 hermitian[] = {z(2, 0), z(1, -1), z(1, 1), z(3, 0)};
    double eigenvalues[2]{};
    metal_complex64 eigenvectors[4]{};
    assert(metal_backend_hermitian_eigh(backend, 2, hermitian, 2, eigenvalues, eigenvectors, 2) == METAL_BACKEND_OK);
    assert(eigenvalues[0] <= eigenvalues[1]);
    assert(std::abs(eigenvalues[0] - 1.0) < 1e-10);
    assert(std::abs(eigenvalues[1] - 4.0) < 1e-10);

    // When a real Metal device is visible, exercise the GPU Jacobi path and
    // the generalized Hermitian-definite path used by Davidson/ZHEGV.
    if (metal_backend_gpu_enabled(backend)) {
        double gpu_w[2]{};
        metal_complex64 gpu_v[4]{};
        const int gpu_rc = metal_backend_hermitian_eigh_gpu(backend, 2, hermitian, 2, gpu_w, gpu_v, 2);
        assert(gpu_rc == METAL_BACKEND_OK);
        assert(std::abs(gpu_w[0] - eigenvalues[0]) < 2e-5);
        assert(std::abs(gpu_w[1] - eigenvalues[1]) < 2e-5);
        for (size_t col = 0; col < 2; ++col) {
            std::complex<double> norm{0.0, 0.0};
            for (size_t row = 0; row < 2; ++row) {
                const std::complex<double> x{gpu_v[row + col * 2].real, gpu_v[row + col * 2].imag};
                norm += std::conj(x) * x;
            }
            assert(std::abs(norm.real() - 1.0) < 2e-4);
        }
        // Fixed-seed complex matrices exercise phase and generalized metric.
        using cd = std::complex<double>;
        std::vector<size_t> dims={1u,2u,4u,8u,16u,32u,40u};
        if(std::getenv("METAL_TEST_LARGE_SUBSPACE")) dims.insert(dims.end(),{48u,64u,128u});
        for (size_t dim : dims) {
            std::mt19937 rng(static_cast<unsigned>(20261002 + dim));
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<metal_complex64> aa(dim*dim), bb(dim*dim), vv(dim*dim), rv(dim*dim);
            std::vector<double> ww(dim), rw(dim);
            for(size_t j=0;j<dim;++j) for(size_t i=0;i<=j;++i) {
                cd x{dist(rng), i==j ? 0.0 : dist(rng)};
                aa[i+j*dim]={x.real(),x.imag()}; aa[j+i*dim]={x.real(),-x.imag()};
                cd y{(i==j ? double(dim)+2.0 : 0.1*dist(rng)), i==j ? 0.0 : 0.1*dist(rng)};
                bb[i+j*dim]={y.real(),y.imag()}; bb[j+i*dim]={y.real(),-y.imag()};
            }
            assert(metal_backend_hermitian_eigh(backend,dim,aa.data(),dim,rw.data(),rv.data(),dim)==METAL_BACKEND_OK);
            assert(metal_backend_hermitian_eigh_gpu(backend,dim,aa.data(),dim,ww.data(),vv.data(),dim)==METAL_BACKEND_OK);
            for(size_t i=0;i<dim;++i) assert(std::abs(ww[i]-rw[i])<1e-10);
            double residual=0.0, ortho=0.0;
            for(size_t j=0;j<dim;++j) for(size_t i=0;i<dim;++i) {
                cd av=0.0, dot=0.0;
                for(size_t k=0;k<dim;++k) {
                    av+=cd(aa[i+k*dim].real,aa[i+k*dim].imag)*cd(vv[k+j*dim].real,vv[k+j*dim].imag);
                    dot+=std::conj(cd(vv[k+i*dim].real,vv[k+i*dim].imag))*cd(vv[k+j*dim].real,vv[k+j*dim].imag);
                }
                residual=std::max(residual,std::abs(av-ww[j]*cd(vv[i+j*dim].real,vv[i+j*dim].imag)));
                ortho=std::max(ortho,std::abs(dot-(i==j ? 1.0 : 0.0)));
            }
            std::printf("eigh n=%zu residual=%.3e orthogonality=%.3e solver=%d iterations=%llu\n",dim,residual,ortho,metal_backend_last_eigh_solver(backend),(unsigned long long)metal_backend_last_eigh_iterations(backend));
            assert(residual<1e-10 && ortho<1e-10);
            for(char triangle : {'U','L'}) {
                assert(metal_backend_generalized_eigh_gpu(backend,1,triangle,'V',dim,aa.data(),dim,bb.data(),dim,ww.data(),vv.data(),dim)==METAL_BACKEND_OK);
                assert(metal_backend_last_gpu_success(backend));
                assert(metal_backend_generalized_all_gpu(backend));
                double gres=0.0, bnorm=0.0;
                std::vector<cd> bv(dim*dim);
                for(size_t j=0;j<dim;++j) for(size_t i=0;i<dim;++i)
                    for(size_t k=0;k<dim;++k) bv[i+j*dim]+=cd(bb[i+k*dim].real,bb[i+k*dim].imag)*cd(vv[k+j*dim].real,vv[k+j*dim].imag);
                for(size_t j=0;j<dim;++j) for(size_t i=0;i<dim;++i) {
                    cd ax=0.0,bx=bv[i+j*dim],metric=0.0;
                    for(size_t k=0;k<dim;++k) {
                        ax+=cd(aa[i+k*dim].real,aa[i+k*dim].imag)*cd(vv[k+j*dim].real,vv[k+j*dim].imag);
                        metric+=std::conj(cd(vv[k+i*dim].real,vv[k+i*dim].imag))*bv[k+j*dim];
                    }
                    gres=std::max(gres,std::abs(ax-ww[j]*bx));
                    bnorm=std::max(bnorm,std::abs(metric-(i==j?1.0:0.0)));
                }
                std::printf("generalized n=%zu uplo=%c residual=%.3e B-orthogonality=%.3e\n",dim,triangle,gres,bnorm);
                assert(gres<1e-10 && bnorm<1e-10);
            }
        }
        // Exact multiplicity and tiny spectral gaps: analytic diagonal spectra.
        for(double gap : {0.0,1e-9}) {
            constexpr size_t dim=12;
            std::vector<metal_complex64> aa(dim*dim),vv(dim*dim);std::vector<double> ww(dim);
            for(size_t i=0;i<dim;++i) aa[i+i*dim]={1.0+gap*i,0.0};
            assert(metal_backend_hermitian_eigh_gpu(backend,dim,aa.data(),dim,ww.data(),vv.data(),dim)==METAL_BACKEND_OK);
            assert(metal_backend_last_gpu_success(backend));
            double residual=0.,ortho=0.;
            for(size_t j=0;j<dim;++j) {
                assert(std::abs(ww[j]-(1.0+gap*j))<1e-10);
                for(size_t i=0;i<dim;++i) {
                    cd dot=0.;for(size_t k=0;k<dim;++k) dot+=std::conj(cd(vv[k+i*dim].real,vv[k+i*dim].imag))*cd(vv[k+j*dim].real,vv[k+j*dim].imag);
                    ortho=std::max(ortho,std::abs(dot-(i==j ? 1.0:0.0)));
                    residual=std::max(residual,std::abs((1.0+gap*i-ww[j])*cd(vv[i+j*dim].real,vv[i+j*dim].imag)));
                }
            }
            printf("clustered spectrum gap=%.3e residual=%.3e orthogonality=%.3e iterations=%llu\n",gap,residual,ortho,(unsigned long long)metal_backend_last_eigh_iterations(backend));
            assert(residual<1e-10 && ortho<1e-10);
        }
        const metal_complex64 ga[] = {z(2, 0), z(0, 0), z(0, 0), z(6, 0)};
        const metal_complex64 gb[] = {z(1, 0), z(0, 0), z(0, 0), z(2, 0)};
        double gw[2]{};
        metal_complex64 gv[4]{};
        const int grc = metal_backend_generalized_eigh_gpu(backend, 1, 'U', 'V', 2, ga, 2, gb, 2, gw, gv, 2);
        assert(grc == METAL_BACKEND_OK);
        assert(std::abs(gw[0] - 2.0) < 2e-5);
        assert(std::abs(gw[1] - 3.0) < 2e-5);
        for (size_t col = 0; col < 2; ++col) {
            std::complex<double> norm{0.0, 0.0};
            for (size_t row = 0; row < 2; ++row) {
                const std::complex<double> x{gv[row + col * 2].real, gv[row + col * 2].imag};
                const std::complex<double> y{gb[row + row * 2].real, gb[row + row * 2].imag};
                norm += std::conj(x) * y * x;
            }
            assert(std::abs(norm.real() - 1.0) < 2e-4);
        }
    }

    metal_backend_destroy(backend);
    return 0;
}
