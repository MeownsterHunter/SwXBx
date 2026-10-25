#!/bin/sh
# Build the library and the example host, then run the host.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
cargo build --quiet
out=target/examples-c
mkdir -p "$out"
${CC:-cc} -Wall -Wextra -Werror -std=c11 -Iinclude examples/host.c \
    -Ltarget/debug -lswxbx -o "$out/host"
LD_LIBRARY_PATH=target/debug "$out/host" "$out/census.txt"
cat "$out/census.txt"
