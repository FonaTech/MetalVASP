// CPU-only policy regression. This does not create or emulate a GPU device.
#include "metal_compile_policy.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    setenv("VASP_METAL_PROFILE", "auto", 1);
    setenv("VASP_METAL_LANGUAGE_VERSION", "2.3", 1);
    auto versions = metal_policy::languages(nil);
    assert(!versions.empty());
    assert(metal_policy::language_name(versions.front()) == "2.3");
    for (auto value : versions)
        assert(static_cast<unsigned long>(value) <= ((2u << 16) + 3));
    setenv("VASP_METAL_PROFILE", "portable", 1);
    versions = metal_policy::languages(nil);
    assert(versions.size() == 1);
    assert(metal_policy::language_name(versions.front()) == "2.3");
    setenv("VASP_METAL_LANGUAGE_VERSION", "auto", 1);
    versions = metal_policy::languages(nil);
    assert(versions.size() == 1 && versions.front() == metal_policy::baseline_msl());
    setenv("VASP_METAL_LANGUAGE_VERSION", "4.0", 1);
    versions = metal_policy::languages(nil);
    assert(versions.size() == 1 && versions.front() == metal_policy::baseline_msl());
    assert(!metal_policy::advanced(nil));
    assert(!metal_policy::metal4(nil));
    std::cout << "compile policy baseline " << metal_policy::language_name(versions.front())
              << "; explicit MSL2.3 ceiling and portable selection passed\n";
}
