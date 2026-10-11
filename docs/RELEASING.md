# Source-only GitHub release

Upload the contents of this release directory to the project repository root. Confirm authorship and third-party boundaries, and keep the `0.15.0-rc3` prerelease label. Keep the private research workspace out of the public repository.

Run `python3 Scripts/audit-public-package.py .` and `python3 -m unittest discover -s Tests -p test_installer.py -v`. Verify archive checksums from the separate `SHA256SUMS` file. Read `release-validation.json`: reviewed backend/native evidence is distinct from the new installer's source-level checks. Full native wrapper builds and the active 20-system campaign are outstanding, so this release should be marked **pre-release**, with those limits visible.

Use the English README as the default GitHub landing page and link the Chinese README. Keep all seven figure families and their PDF/SVG/CSV/JSON companions. Cite the actual tag/commit and archive SHA; replace no missing DOI with an invented identifier. GitHub CI checks the installer and public package without VASP fixtures, secrets or licensed downloads.

Directory names are case-sensitive on Linux: keep `Tests/` exactly as packaged. Both CI matrix jobs run independently with `fail-fast: false`. The audit checks test discovery against actual filename spelling, even on a case-insensitive Mac, and rejects a selection containing no test files.

Do not attach privately modified VASP trees, POTCAR, linked VASP executables, dependency libraries or private logs. MIT does not authorize them. For another VASP version, follow the compatibility contract before advertising support. Release binaries only if every separate license/dependency/runtime requirement has been independently resolved; no binaries are supplied here.

As more cases complete, review their CPU/GPU input, full-SCF, timing and resource gates before extending the public snapshot. Export only the released GPU route versus CPU; running cases are not measurements. Preserve source hashes and missing/slow results.
