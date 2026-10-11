#!/usr/bin/env python3
"""Compile exactly the backend's concatenated shader strings, with provenance."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

PARTS = [
    ("metal_backend.mm", "kBackendShader"),
    ("metal_backend.mm", "kResidentShader"),
    ("lobpcg_shader.h", "kLobpcgShader"),
    ("subspace_shader.h", "kSubspaceShader"),
    ("subspace_shader.h", "kCyclicShader"),
    ("hamiltonian_shader.h", "hamiltonian_shader"),
    ("orthogonal_shader.h", "orthogonal_shader"),
    ("wave_owner_shader.h", "wave_owner_shader"),
    ("nonlocal_shader.h", "nonlocal_shader"),
]
SAFE_MATH_FLAGS = [
    "-fmetal-math-mode=safe",
    "-fmetal-math-fp32-functions=precise",
    "-ffp-contract=off",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(project):
    pieces, sources = [], []
    backend = (project / "Sources" / "metal_backend.mm").read_text()
    assembly = re.search(r"\bNSString\s*\*\s*source\s*=", backend)
    if not assembly:
        raise ValueError("Cannot find backend runtime shader assembly")
    finish = backend.find("auto compiled=", assembly.start())
    if finish < 0:
        raise ValueError("Cannot find runtime shader compilation after assembly")
    actual = re.findall(r"stringWithUTF8String:\s*(\w+)", backend[assembly.start():finish])
    if actual != [name for _, name in PARTS]:
        raise ValueError(f"Runtime shader assembly changed; update the offline parts list: {actual}")
    for filename, name in PARTS:
        path = project / "Sources" / filename
        data = path.read_bytes()
        pattern = rf"\b{re.escape(name)}\s*(?:\[\s*\])?\s*=\s*R\"([^\s()\\]*)\((.*?)\)\1\""
        found = re.findall(pattern, data.decode(), re.DOTALL)
        if len(found) != 1:
            raise ValueError(f"Expected exactly one raw shader string {name} in {path}")
        piece = found[0][1]
        pieces.append(piece)
        sources.append({"path": str(path), "symbol": name, "file_sha256": sha(data),
                        "shader_sha256": sha(piece.encode())})
    source = "".join(pieces)
    if "#pragma METAL fp contract(off)" not in source:
        raise ValueError("Backend DS contraction guard is missing")
    return source, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--language", choices=["2.3", "2.4", "3.0", "3.1", "3.2", "4.0"], default="4.0")
    parser.add_argument("--device-tuned", choices=[0, 1], type=int, default=0)
    parser.add_argument("--metal4", choices=[0, 1], type=int, default=0)
    parser.add_argument("--deployment-target", help="Actual macOS compiler minimum; defaults to 13.0 for baseline, or the selected language's minimum")
    parser.add_argument("--compiler-flag", action="append", choices=SAFE_MATH_FLAGS, default=[], help="Verified safe math option; pass all three as --compiler-flag=-option")
    parser.add_argument("--emit-only", action="store_true", help="Extract source/provenance without claiming compilation")
    args = parser.parse_args()
    deployment = args.deployment_target or {"3.1":"14.0", "3.2":"15.0", "4.0":"26.0"}.get(args.language, "13.0")
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", deployment):
        parser.error("Deployment target must be a numeric macOS version, such as 13.0")
    project, output = args.project.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"Output exists: {output}; select a new directory to preserve evidence")
    source, sources = extract(project)
    # The compiler's macOS Metal 2.x dialects retain a platform prefix;
    # Metal 3.x and 4.x use the unprefixed names in its diagnostic/help.
    standard = ("macos-metal" if args.language.startswith("2.") else "metal") + args.language
    if not args.emit_only and (len(args.compiler_flag) != len(SAFE_MATH_FLAGS) or set(args.compiler_flag) != set(SAFE_MATH_FLAGS)):
        parser.error("Pass each verified safe/precise/no-contraction math flag exactly once")
    output.mkdir(parents=True)
    shader, air, library = output / "backend.metal", output / "backend.air", output / "backend.metallib"
    shader.write_text(source)
    manifest = {"source_parts": sources, "runtime_source_assembly_verified": True,
                "shader_sha256": sha(source.encode()), "language": args.language,
                "compiler_standard": standard,
                "deployment_target": deployment,
                "macros": {"METAL_VASP_DEVICE_TUNED": args.device_tuned, "METAL_VASP_METAL4": args.metal4},
                "compiler_flags": args.compiler_flag,
                "kernel_names": sorted(set(re.findall(r"\bkernel\s+void\s+(\w+)\s*\(", source))),
                "compilation_success": False, "gpu_validation": "not-run"}

    def save():
        (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.emit_only:
        manifest["status"] = "source-extracted-only"
        save()
        print(json.dumps({"output": str(output), "status": manifest["status"], "kernel_count": len(manifest["kernel_names"])}))
        return 0
    commands = [
        ["xcrun", "metal", "-v"],
        ["xcrun", "metal", f"-std={standard}", f"-mmacosx-version-min={deployment}", *args.compiler_flag,
         f"-DMETAL_VASP_DEVICE_TUNED={args.device_tuned}", f"-DMETAL_VASP_METAL4={args.metal4}",
         "-c", str(shader), "-o", str(air)],
        ["xcrun", "metallib", str(air), "-o", str(library)],
    ]
    manifest["commands"] = []
    for index, command in enumerate(commands):
        completed = subprocess.run(command, capture_output=True, text=True)
        (output / f"command-{index}.stdout.log").write_text(completed.stdout)
        (output / f"command-{index}.stderr.log").write_text(completed.stderr)
        manifest["commands"].append({"argv": command, "returncode": completed.returncode})
        if completed.returncode:
            manifest["status"] = "compilation-failed"
            save()
            print(json.dumps({"output": str(output), "status": manifest["status"], "failed_command": index}))
            return completed.returncode
    manifest.update({"status": "offline-compilation-succeeded", "compilation_success": True,
                     "air_sha256": sha(air.read_bytes()), "metallib_sha256": sha(library.read_bytes())})
    save()
    print(json.dumps({"output": str(output), "library": str(library), "metallib_sha256": manifest["metallib_sha256"],
                      "kernel_count": len(manifest["kernel_names"]), "gpu_validation": "not-run"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
