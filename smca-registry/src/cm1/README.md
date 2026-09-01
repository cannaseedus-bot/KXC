# CM-1 Verifier

The CM-1 verifier is compiled to WebAssembly and exposes a single export:

- `verify(bytes, length)` → `1` for legal CM-1 streams, `0` otherwise.

Legal CM-1 streams follow the grammar:

```
SOH STX (SO RS SI)* ETX EOT
```
