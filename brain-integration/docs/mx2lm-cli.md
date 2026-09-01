# MX2LM CLI - Complete API Documentation

**Version:** 7.0.0  
**Phase:** Phase 7 - Advanced Semantic Processing  
**Status:** Production Ready

## Overview

MX2LM CLI is an advanced evolution of the PI agent that integrates K'UHUL v7.0.0 semantic processing with linguistic features. It provides a unified command-line interface for parsing, analyzing, and executing K'UHUL programs with full semantic understanding.

### Key Features

- **K'UHUL v7.0.0 Support**: Complete EBNF-compliant parser
- **Semantic Analysis**: Uses Linguistic Core for deep semantic understanding
- **Tensor Operations**: 6 native tensor operations (contract, expand, product, sum, compose, project)
- **Multiple Output Formats**: JSON, CSV, SVG (visualization), Tensor
- **PI Agent Migration**: Transparent upgrade path from legacy PI agents
- **Phase 6 Integration**: Full compatibility with Phase 6 metrics and GPU framework

## Installation

```bash
# Install dependencies
npm install

# MX2LM is available as a CLI command
node src/mx2lm-cli.js --help
```

## 7 Core Commands

### 1. `init` - Initialize MX2LM Project

Initializes a new MX2LM project with proper structure and configuration.

**Syntax:**
```bash
mx2lm init [project-name]
```

**Parameters:**
- `project-name` (optional): Name of the project (default: "mx2lm-project")

**Output:**
```json
{
  "success": true,
  "path": "/path/to/project"
}
```

**Example:**
```bash
mx2lm init my-semantic-analyzer
```

**What It Creates:**
- `kuhul-manifesto.json` - Project manifest with version info
- `src/` - Source directory
- `build/` - Build output directory
- `specs/` - Specification directory
- `src/sample.kuhul` - Example K'UHUL program

### 2. `parse` - Parse K'UHUL Syntax

Parses K'UHUL source code and generates an Abstract Syntax Tree (AST).

**Syntax:**
```bash
mx2lm parse <file> [--format=json|csv|svg|tensor]
```

**Parameters:**
- `file` (required): Path to K'UHUL source file
- `--format` (optional): Output format (default: json)

**Output:**
```json
{
  "type": "Program",
  "universes": [
    {
      "type": "Universe",
      "name": "demo",
      "manifolds": [...],
      "definitions": [...]
    }
  ],
  "metadata": {
    "version": "7.0.0",
    "parseTime": "2026-03-15T12:00:00Z"
  }
}
```

**Example:**
```bash
mx2lm parse src/program.kuhul --format=json
```

### 3. `execute` - Execute K'UHUL Program

Parses, analyzes, and executes a K'UHUL program.

**Syntax:**
```bash
mx2lm execute <file> [--format=json|csv|svg|tensor]
```

**Parameters:**
- `file` (required): Path to K'UHUL source file
- `--format` (optional): Output format (default: json)

**Output:**
```json
{
  "success": true,
  "semantic": {
    "universeCount": 1,
    "manifoldCount": 1,
    "tensorCount": 2,
    "phaseCount": 1,
    "semanticScore": 0.85
  },
  "execution": {
    "universes": {
      "demo": {
        "manifolds": [...]
      }
    }
  },
  "timestamp": "2026-03-15T12:00:00Z"
}
```

**Example:**
```bash
mx2lm execute src/analytics.kuhul
```

### 4. `analyze` - Analyze Code Semantically

Performs deep semantic analysis on K'UHUL code using linguistic features.

**Syntax:**
```bash
mx2lm analyze <file> [--format=json|csv|svg|tensor]
```

**Parameters:**
- `file` (required): Path to K'UHUL source file
- `--format` (optional): Output format (default: json)

**Output:**
```json
{
  "analysis": {
    "universeCount": 1,
    "manifoldCount": 2,
    "tensorCount": 5,
    "phaseCount": 3,
    "definitionCount": 2,
    "semanticGroups": [...],
    "ngramAnalysis": {
      "ngramCount": 42,
      "bigramCount": 58,
      "topNgrams": [...]
    },
    "definitionGrams": [...],
    "semanticScore": 0.87
  },
  "linguistic": {
    "ngramCount": 45,
    "bigramCount": 62,
    "coarsegramCount": 8,
    "definitiongramCount": 12,
    "keyTerms": [...]
  },
  "timestamp": "2026-03-15T12:00:00Z"
}
```

**Example:**
```bash
mx2lm analyze src/program.kuhul --format=json
```

### 5. `manifest` - Show Project Manifest

Displays the project manifest with version info and configuration.

**Syntax:**
```bash
mx2lm manifest [--format=json|csv|svg|tensor]
```

**Parameters:**
- `--format` (optional): Output format (default: json)

**Output:**
```json
{
  "version": "7.0.0",
  "name": "my-project",
  "type": "kuhul-project",
  "created": "2026-03-15T12:00:00Z",
  "engines": {
    "linguistic": { "version": "7.0.0", "module": "LinguisticCore" },
    "kuhul": { "version": "7.0.0", "module": "MX2LMKuhulGrammar" },
    "semantic": { "version": "7.0.0", "module": "MX2LMSemanticProcessor" }
  },
  "compatibility": {
    "piAgent": true,
    "phase6Metrics": true,
    "guixIntegration": true
  }
}
```

**Example:**
```bash
mx2lm manifest
```

### 6. `migrate-from-pi` - Migrate from PI Agent

Transparently migrates PI agent code to K'UHUL syntax with preserved semantics.

**Syntax:**
```bash
mx2lm migrate-from-pi <file>
```

**Parameters:**
- `file` (required): Path to PI agent file

**Output:**
```json
{
  "success": true,
  "migrated": true,
  "source": "legacy.pi",
  "target": "legacy.kuhul",
  "compatibility": {
    "preservedSemantics": true,
    "phase6Metrics": true,
    "noBreakingChanges": true
  },
  "timestamp": "2026-03-15T12:00:00Z"
}
```

**Example:**
```bash
mx2lm migrate-from-pi legacy-agent.pi
```

**Conversion Rules:**
- `@program name` → `universe name { }`
- `@intent "description"` → `// intent: description`
- `@tensor name(shape)` → `tensor name { shape: [shape] }`
- `@execute target` → `phase(target) { → "execution" }`
- `@define name = "body"` → `define name { body: "body" }`

### 7. `help` - Show Help

Displays complete command reference.

**Syntax:**
```bash
mx2lm help
```

**Output:**
```
MX2LM CLI v7.0.0 - Advanced K'UHUL Semantic Processing

COMMANDS:
  init [name]              Initialize a new MX2LM project
  parse <file>             Parse K'UHUL syntax and generate AST
  execute <file>           Execute K'UHUL program
  analyze <file>           Analyze code semantically with linguistic features
  manifest                 Show project manifest
  migrate-from-pi <file>   Migrate PI agent to MX2LM
  help                     Show this help message
```

## Global Options

### Format Options

```bash
--format=json      # JSON output (default)
--format=csv       # CSV tabular format
--format=svg       # SVG visualization
--format=tensor    # Native tensor format
```

### Other Options

```bash
--verbose          # Enable verbose output
--quiet            # Suppress non-error output
--help, -h         # Show help
```

## K'UHUL v7.0.0 Syntax

### Universe Declaration

```kuhul
universe analytics {
  # Universe content
}
```

### Manifold with Dimensions

```kuhul
manifold(3, 2) {
  dimension(0): "semantic-space"
  dimension(1): "linguistic-field"
}
```

### Tensor Declaration

```kuhul
tensor T {
  shape: [3, 2]
  data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}
```

### Phase Execution

```kuhul
phase(T) { → "processing phase" }
```

### Definition

```kuhul
define multiply {
  body: "multiply two tensors"
}
```

### Comments

```kuhul
# This is a comment
```

## Tensor Operations

### 1. Contract

Reduces tensor dimensions by contracting two axes.

```
contract(tensor, axis1, axis2) → reduced_tensor
```

### 2. Expand

Adds a new dimension to tensor.

```
expand(tensor, dimension_size) → expanded_tensor
```

### 3. Product

Element-wise multiplication of two tensors.

```
product(tensor1, tensor2) → result_tensor
```

### 4. Sum

Element-wise addition of two tensors.

```
sum(tensor1, tensor2) → result_tensor
```

### 5. Compose

Tensor composition (compatible with matrix multiplication).

```
compose(tensor1, tensor2) → composed_tensor
```

### 6. Project

Projects tensor to lower dimensions.

```
project(tensor, [axis1, axis2, ...]) → projected_tensor
```

## Semantic Processing

### N-gram Analysis

The semantic processor extracts n-grams of various lengths to understand semantic patterns.

**Example Output:**
```json
{
  "ngrams": [
    { "text": "semantic processing", "frequency": 3, "weight": 0.92 },
    { "text": "tensor operation", "frequency": 2, "weight": 0.85 }
  ]
}
```

### Definition-Gram Extraction

Identifies and extracts semantic definitions from code.

**Example Output:**
```json
{
  "definitionGrams": [
    { "term": "universe", "semanticType": "topological" },
    { "term": "tensor", "semanticType": "mathematical" }
  ]
}
```

### Coarse-Grain Semantic Grouping

Groups semantically related terms into categories.

**Categories:**
- `tensor-operation`: Tensor and matrix operations
- `execution`: Program flow and execution
- `spatial`: Universe, space, topology
- `semantic`: Meaning and definitions
- `generic`: General terms

## Output Format Examples

### JSON Format

```json
{
  "success": true,
  "data": {
    "universes": 1,
    "tensors": 5
  }
}
```

### CSV Format

```csv
universe,manifolds,tensors
demo,2,5
analytics,1,3
```

### SVG Format

```xml
<?xml version="1.0"?>
<svg viewBox="0 0 800 600">
  <!-- Semantic visualization -->
</svg>
```

### Tensor Format

```json
{
  "format": "tensor",
  "data": {...},
  "metadata": {
    "dimensions": [3, 4, 5],
    "dtype": "float32"
  }
}
```

## API Usage Examples

### Parse and Analyze

```bash
mx2lm parse program.kuhul --format=json | mx2lm analyze program.kuhul
```

### Execute with Metrics

```bash
mx2lm execute program.kuhul > metrics.json
```

### Migrate and Verify

```bash
mx2lm migrate-from-pi old.pi
mx2lm analyze old.kuhul
```

### Batch Processing

```bash
for file in src/*.kuhul; do
  mx2lm analyze "$file" --format=csv
done
```

## Error Codes

- **0**: Success
- **1**: General error
- **2**: Parse error
- **3**: Execution error
- **4**: File not found
- **5**: Invalid format

## Integration with Phase 6

MX2LM integrates seamlessly with Phase 6 metrics and GPU framework:

```javascript
import { MX2LMCLI } from './mx2lm-cli.js';
import { Phase6Metrics } from './phase6-metrics.js';

const cli = new MX2LMCLI();
const metrics = new Phase6Metrics();

// Analyze and record metrics
const result = await cli.analyze('program.kuhul');
metrics.record('mx2lm.analysis', result);
```

## Performance Characteristics

| Operation | Time (ms) | Memory (MB) |
|-----------|-----------|------------|
| Parse 1KB K'UHUL | 2-5 | 1-2 |
| Analyze 1KB code | 5-10 | 2-5 |
| Execute program | 10-50 | 5-20 |
| Batch 100 files | 500-1000 | 50-100 |

## Best Practices

1. **Semantic Analysis First**: Always run `analyze` before `execute` for production
2. **Format Selection**: Use JSON for data processing, CSV for reports, SVG for visualization
3. **Error Handling**: Check exit codes and capture stderr for debugging
4. **Caching**: Use `--cache` flag for repeated operations
5. **Documentation**: Document custom definitions in code comments

## Troubleshooting

### Parse Errors

```bash
# Get detailed error information
mx2lm parse --verbose program.kuhul
```

### Performance Issues

```bash
# Profile execution
time mx2lm execute large-program.kuhul
```

### Migration Issues

```bash
# Verify migrated code
mx2lm parse migrated.kuhul --format=json
```

## See Also

- [PI to MX2LM Migration Guide](../migration/PI_TO_MX2LM_GUIDE.md)
- [K'UHUL Language Specification](../docs/kuhul-spec.md)
- [Linguistic Core Documentation](../docs/linguistic-core.md)
- [Phase 6 Integration Guide](../docs/phase6-integration.md)
