# Brain Integration

This folder contains stubs for integrating with an external Python runner.

Files:
- bridge.ps1 — PowerShell bridge wrapper exporting start-server/stop-server/send; delegates to bridge_py_runner.py (PowerShell preferred runtime).
- bridge.test.ts — Jest test skeleton
- bridge_py_runner.py — Python runner stub with handle(prompt) function
- README.md — this file

Running tests (TypeScript):
1. From repository root, run: npm test
2. Or run: npx jest src/ts/public/Win2D/brain-integration/bridge.test.ts

Running Python runner:
1. Use: python C:\public_html\MX2LM\codex\AS-XCFE\artifacts\shards\desp-v1\src\ts\public\Win2D\brain-integration\bridge_py_runner.py
2. The runner reads stdin and prints a response; call handle(prompt) within your integration.
