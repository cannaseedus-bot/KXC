#!/usr/bin/env bash
set -e

clang \
  --target=wasm32 \
  -O3 \
  -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=__heap_base \
  -Wl,--export=memory \
  -Wl,--export=verify \
  -Wl,--allow-undefined \
  -o ../../wasm/cm1_verify.wasm \
  cm1_verify_wasm.c
