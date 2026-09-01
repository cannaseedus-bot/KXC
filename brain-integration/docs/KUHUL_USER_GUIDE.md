# K'UHUL User Guide

A practical guide to using K'UHUL v7.0.0 for semantic programming.

## Table of Contents

1. [Getting Started](#getting-started)
2. [CLI Commands](#cli-commands)
3. [REPL Tutorial](#repl-tutorial)
4. [Program Structure](#program-structure)
5. [Common Patterns](#common-patterns)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
npm install kuhul-cli
```

### First Program

Create `hello.kuhul`:

```
let greeting = "Hello, K'UHUL!";
println(greeting);
```

Run it:

```bash
kuhul execute hello.kuhul
```

### Check Version

```bash
kuhul version
```

Output:
```
K'UHUL CLI v7.0.0
Language: K'UHUL v7.0.0
EBNF-compliant parser
Complete semantic analysis and execution engine
```

## CLI Commands

### Core Commands

#### `parse <file>`
Parse a K'UHUL file and check syntax.

```bash
kuhul parse myprogram.kuhul
```

#### `execute <file>`
Parse and execute a K'UHUL program.

```bash
kuhul execute myprogram.kuhul
```

#### `compile <file> [output]`
Compile and optimize a program.

```bash
kuhul compile myprogram.kuhul optimized.kuhul.opt
```

#### `repl`
Start interactive REPL session.

```bash
kuhul repl
```

### Universe Management

#### `universe list`
List all universes.

```bash
kuhul universe list
```

#### `universe create <name> [version]`
Create a new universe.

```bash
kuhul universe create MyUniverse 1.0.0
```

#### `universe delete <name>`
Delete a universe.

```bash
kuhul universe delete MyUniverse
```

#### `universe info [name]`
Show universe information.

```bash
kuhul universe info MyUniverse
```

#### `universe set-default <name>`
Set the default universe.

```bash
kuhul universe set-default MyUniverse
```

### Type System Commands

#### `type infer <expression>`
Infer the type of an expression.

```bash
kuhul type infer "hello"
```

#### `type check <file>`
Type check a file without executing.

```bash
kuhul type check myprogram.kuhul
```

### Tensor Operations

#### `tensor ops <operation> [args...]`
Perform tensor operations.

```bash
kuhul tensor ops matrix 3 3     # Create 3x3 matrix
kuhul tensor ops vector 5       # Create 5D vector
```

### Agent Management

#### `agent list`
List agents in current cluster.

```bash
kuhul agent list
```

### Manifold Operations

#### `manifold list`
List manifolds in current universe.

```bash
kuhul manifold list
```

#### `manifold create <name> [type] [dimensions]`
Create a new manifold.

```bash
kuhul manifold create space euclidean 3
kuhul manifold create spacetime minkowski 4
kuhul manifold create curved riemannian 3
```

Manifold types:
- `euclidean`: Flat Euclidean geometry
- `riemannian`: Curved geometry
- `minkowski`: Relativistic spacetime
- `pi-harmonic`: π-harmonic geometry

### Help System

#### `help`
Show all available commands.

```bash
kuhul help
```

#### `help <command>`
Show help for specific command.

```bash
kuhul help execute
```

## REPL Tutorial

Start the REPL:

```bash
kuhul repl
```

### Basic Expressions

```
kuhul> 2 + 3
=> 5
kuhul> "hello" + " world"
=> "hello world"
kuhul> true && false
=> false
```

### Variables

```
kuhul> let x = 42
kuhul> x
=> 42
kuhul> let y = x * 2
kuhul> y
=> 84
```

### Functions

```
kuhul> func add(a, b) { a + b }
kuhul> add(3, 4)
=> 7
```

### Arrays

```
kuhul> let arr = [1, 2, 3]
kuhul> arr
=> [1, 2, 3]
kuhul> length(arr)
=> 3
```

### Objects

```
kuhul> let person = { name: "Alice", age: 30 }
kuhul> person
=> { name: 'Alice', age: 30 }
```

### Type Inspection

```
kuhul> type_of(42)
=> "int"
kuhul> type_of("hello")
=> "string"
kuhul> type_of([1, 2, 3])
=> "tuple"
```

### Universe Operations

```
kuhul> universe list
Universes:
  * default (v1.0.0)

kuhul> universe create test 1.0.0
✓ Universe 'test' created

kuhul> universe set-default test
✓ Default universe set to 'test'
```

### Manifold Creation

```
kuhul> manifold create space euclidean 3
✓ Manifold 'space' created (euclidean, 3D)

kuhul> manifold list
Manifolds:
  space: euclidean (3D)
```

### Shell Commands

Execute shell commands with `!`:

```
kuhul> !ls
(lists files)

kuhul> !date
(shows current date)
```

### Exit

```
kuhul> exit
Goodbye!

kuhul> quit
Goodbye!
```

## Program Structure

### Basic Layout

```
// Universe definition
universe MyApp : "1.0.0" {
  // Type definitions
  type Point3D = point;
  
  // Function definitions
  func distance(p1: point, p2: point) -> float {
    // Implementation
  }
  
  // Main logic
  let origin = { x: 0, y: 0, z: 0 };
}
```

### Comments

```
// Single-line comment

/* Multi-line
   comment */
```

### Module-like Organization

```
// utils.kuhul
func add(a, b) -> int {
  a + b
}

func multiply(a, b) -> int {
  a * b
}

// main.kuhul (can import utils conceptually)
let result = add(5, 3);
println(result);
```

## Common Patterns

### Mathematical Functions

```
func square(x: float) -> float {
  x ^ 2
}

func cube(x: float) -> float {
  x ^ 3
}

func quadratic(a: float, b: float, c: float, x: float) -> float {
  a * (x ^ 2) + b * x + c
}
```

### List Processing

```
func sum(arr: any) -> float {
  let result = 0;
  for (let i = 0; i < length(arr); i += 1) {
    result = result + at(arr, i);
  }
  return result;
}

func average(arr: any) -> float {
  sum(arr) / length(arr);
}
```

### Conditional Logic

```
func absolute(x: float) -> float {
  if (x < 0) {
    return -x;
  } else {
    return x;
  }
}

func max(a: float, b: float) -> float {
  if (a > b) a else b
}
```

### Recursion

```
func factorial(n: int) -> int {
  if (n <= 1) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

func fibonacci(n: int) -> int {
  if (n <= 1) {
    return n;
  } else {
    return fibonacci(n - 1) + fibonacci(n - 2);
  }
}
```

### Matrix Operations

```
func identity(n: int) -> matrix {
  // Create n×n identity matrix
  let mat = matrix(n, n);
  for (let i = 0; i < n; i += 1) {
    mat[i, i] = 1;
  }
  return mat;
}
```

### Type-Safe Code

```
type UserId = int;
type Age = int;

func validateAge(age: Age) -> bool {
  age > 0 && age < 150
}

func registerUser(id: UserId, age: Age) -> bool {
  validateAge(age)
}
```

## Advanced Features

### Phase Space Operations

```
// Create phase states
let state1 = phase_push({ x: 0.5, y: 0.3 });
let state2 = phase_push({ x: 0.7, y: 0.2 });

// Create transition
let trans = phase_transition(state1, state2);
```

### Agent Coordination

```
func worker(id: int) {
  println("Worker " + id + " starting");
  // Do work
  println("Worker " + id + " done");
}

// Spawn agent
let agent = spawn(func { worker(1) });
join(agent);  // Wait for completion
```

### Tensor Contractions

```
func trace(m: matrix) -> float {
  // Sum diagonal elements
  trace(m)
}

func frobenius(m: matrix) -> float {
  // Frobenius norm
  sqrt(trace(transpose(m) * m))
}
```

### Custom Universe

```
universe Analytics : "1.0.0" {
  manifold metrics: euclidean(10) {
    // 10-dimensional metrics space
  }
  
  type MetricsPoint = point;
  type MetricsVector = vector;
  
  func distance(p1: MetricsPoint, p2: MetricsPoint) -> float {
    // Calculate distance
  }
}
```

## Best Practices

### 1. Type Annotations

Always annotate function parameters and return types:

```
// Good
func add(a: int, b: int) -> int {
  a + b
}

// Avoid
func add(a, b) {
  a + b
}
```

### 2. Meaningful Names

Use clear, descriptive names:

```
// Good
func calculateAverageScore(scores: any) -> float {
  sum(scores) / length(scores)
}

// Avoid
func calc(s: any) -> float {
  s / length(s);  // Wrong!
}
```

### 3. Function Decomposition

Break complex logic into smaller functions:

```
// Good
func validateEmail(email: string) -> bool {
  hasAtSign(email) && hasValidDomain(email)
}

// Avoid
func validateEmail(email: string) -> bool {
  email != "" && email[indexOf(email, "@")] == "@" && ...
}
```

### 4. Error Handling

Plan for edge cases:

```
func divide(a: float, b: float) -> float {
  if (b == 0) {
    return 0;  // or return error
  }
  return a / b;
}
```

### 5. Documentation

Add comments for non-obvious logic:

```
func quicksort(arr: any) -> any {
  // Partition array around pivot
  // Recursively sort left and right
  if (length(arr) <= 1) {
    return arr;
  }
  // ... implementation
}
```

### 6. Immutability

Prefer creating new values:

```
// Good
func increment(arr: any) -> any {
  let result = [];
  for (let i = 0; i < length(arr); i += 1) {
    push(result, at(arr, i) + 1);
  }
  return result;
}
```

### 7. Performance

Avoid unnecessary allocations:

```
// Good
func fastSum(arr: any) -> float {
  let sum = 0;
  for (let i = 0; i < length(arr); i += 1) {
    sum = sum + at(arr, i);
  }
  return sum;
}
```

## Troubleshooting

### Common Errors

#### "Unexpected token in expression"

Usually means a syntax error. Check:
- Matching parentheses and braces
- Proper operator usage
- Correct statement structure

```
// Wrong
let x = 5 +;  // Missing operand

// Correct
let x = 5 + 3;
```

#### "Expected identifier"

Often a parsing error. Check:
- Variable names after `let`
- Function names after `func`
- Key names in objects

```
// Wrong
let 123 = 5;  // Variable name can't be number

// Correct
let x123 = 5;
```

#### "Type mismatch"

Type incompatibility. Check:
- Operand types match operator requirements
- Function arguments match parameter types

```
// Wrong
let x: int = "hello";  // Can't assign string to int

// Correct
let x: string = "hello";
```

### Debugging Tips

1. **Use type check command**:
   ```bash
   kuhul type check myprogram.kuhul
   ```

2. **Add debugging prints**:
   ```
   println("Debug: x = " + x);
   ```

3. **Test in REPL first**:
   ```bash
   kuhul repl
   kuhul> let x = 5;
   kuhul> x + 3
   ```

4. **Parse without execution**:
   ```bash
   kuhul parse myprogram.kuhul
   ```

5. **Check manifold setup**:
   ```bash
   kuhul manifold list
   ```

### Performance Optimization

1. Use vectorized operations where possible
2. Avoid deep recursion
3. Cache computed results
4. Use type constraints for validation

## Additional Resources

- **Language Reference**: kuhul-language.md
- **Examples**: See examples/ directory
- **CLI Help**: `kuhul help`
- **REPL**: `kuhul repl`

## Contributing

Report issues and suggest improvements to the K'UHUL project.
