# Binary artifacts and examples

This repository avoids committing binary artifacts to keep GitHub uploads predictable.
Regenerate them locally as needed.

## CM-1 WASM verifier

Build the CM-1 verifier from source:

```bash
cd src/cm1
./build.sh
```

This creates `wasm/cm1_verify.wasm`, which is required by `node conformance/run.js`.

## Example CM-1 streams

If you need example CM-1 streams as files, create them with the following contents:

**valid.cm1**
```
01 02 0E 1E 0F 0E 1E 0F 03 04
```

**invalid.cm1**
```
01 02 0E 03 04
```
