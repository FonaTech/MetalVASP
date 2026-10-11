#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$root/Scripts/toolchain-env.sh"
exec python3 "$root/Scripts/metalvasp.py" "$@"
