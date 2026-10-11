#!/bin/sh
# Sourced by build scripts. Preserve explicitly selected tools and make the
# native Homebrew installation discoverable when a shell has no brew setup.
case $(uname -m) in
    arm64) metal_brew_bin1=/opt/homebrew/bin; metal_brew_bin2=/usr/local/bin;;
    *) metal_brew_bin1=/usr/local/bin; metal_brew_bin2=/opt/homebrew/bin;;
esac
for metal_brew_bin in "$metal_brew_bin1" "$metal_brew_bin2"; do
    if [ -d "$metal_brew_bin" ]; then
        case ":$PATH:" in *":$metal_brew_bin:"*) ;; *) PATH="$PATH:$metal_brew_bin";; esac
    fi
done
export PATH
unset metal_brew_bin1 metal_brew_bin2 metal_brew_bin
