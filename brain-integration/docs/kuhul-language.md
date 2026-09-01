# K'UHUL v7.0.0 Language Reference

Complete language specification for K'UHUL, the semantic language for quantum-geometric computing.

## Table of Contents

1. [Overview](#overview)
2. [EBNF Grammar](#ebnf-grammar)
3. [Type System](#type-system)
4. [Operators](#operators)
5. [Control Flow](#control-flow)
6. [Functions](#functions)
7. [Universes and Manifolds](#universes-and-manifolds)
8. [Type Constraints](#type-constraints)
9. [Built-in Functions](#built-in-functions)
10. [Examples](#examples)

## Overview

K'UHUL is a semantic language designed for expressing geometric and quantum computational concepts. It provides:

- **Universes**: Top-level containers for logical scopes with versioning
- **Manifolds**: Geometric execution spaces (Euclidean, Riemannian, Minkowski, π-harmonic)
- **Phase Spaces**: Quantum field execution substrate with resolution bits
- **Agent Clusters**: Multi-agent coordination with shared memory
- **Type System**: 30+ types covering primitives, geometrics, and phase operations
- **Semantic Analysis**: Full type inference and checking

## EBNF Grammar

```ebnf
program         = { statement };
statement       = universeDecl | funcDecl | typeDecl | varDecl | ifStmt | 
                  whileStmt | forStmt | returnStmt | exprStmt ;
universeDecl    = "universe" identifier [ ":" expression ] [ block ] ;
funcDecl        = "func" identifier "(" [ parameters ] ")" [ "->" type ] [ block ] ;
typeDecl        = "type" identifier "=" type ;
varDecl         = "let" identifier [ ":" type ] [ "=" expression ] ;
ifStmt          = "if" "(" expression ")" statement [ "else" statement ] ;
whileStmt       = "while" "(" expression ")" statement ;
forStmt         = "for" "(" [ varDecl | expression ] ";" [ expression ] ";" [ expression ] ")" statement ;
returnStmt      = "return" [ expression ] ;
exprStmt        = expression [ ";" ] ;
block           = "{" { statement } "}" ;

expression      = assignment ;
assignment      = ternary [ ( "=" | "+=" | "-=" | "*=" | "/=" ) assignment ] ;
ternary         = logicalOr [ "?" expression ":" expression ] ;
logicalOr       = logicalAnd { "||" logicalAnd } ;
logicalAnd      = equality { "&&" equality } ;
equality        = comparison { ( "==" | "!=" ) comparison } ;
comparison      = bitwise { ( "<" | ">" | "<=" | ">=" ) bitwise } ;
bitwise         = additive { ( "&" | "|" | "^" ) additive } ;
additive        = multiplicative { ( "+" | "-" ) multiplicative } ;
multiplicative  = exponentiation { ( "*" | "/" | "%" ) exponentiation } ;
exponentiation  = unary [ "^" exponentiation ] ;
unary           = [ ( "!" | "-" | "+" | "~" | "async" | "await" ) ] postfix ;
postfix         = primary { ( "(" [ args ] ")" | "[" expression "]" | "." identifier ) } ;
primary         = number | string | bool | identifier | "(" expression ")" | 
                  "[" [ expressions ] "]" | "{" [ pairs ] "}" ;

type            = typeBase [ "?" | ( "[" "]" ) ] ;
typeBase        = identifier | "(" [ types ] ")" "->" type ;
```

## Type System

### Primitive Types

- `bool`: Boolean (true/false)
- `int`: Integer (-2^63 to 2^63-1)
- `float`: Floating-point number (IEEE 754 double)
- `string`: Text string (UTF-8)
- `bytes`: Binary data
- `null`: Null/void type

### Geometric Types

- `point`: N-dimensional point in manifold
- `vector`: N-dimensional vector
- `matrix`: M×N matrix
- `tensor`: N-dimensional tensor

### Phase Types

- `phase_state`: Quantum phase state with resolution bits
- `phase_transition`: Transition between phase states

### Compound Types

- `tuple`: Ordered collection of heterogeneous values `(T1, T2, ...)`
- `record`: Named collection `{field1: T1, field2: T2, ...}`
- `union`: Tagged union type `T1 | T2 | ...`
- `option`: Optional value `T?`

### Type Aliases

Create custom types:
```
type Point3D = point;
type Matrix4x4 = matrix;
type StateSet = (phase_state, phase_state)?;
```

## Operators

### Arithmetic Operators

- `+`: Addition (int, float, string concatenation)
- `-`: Subtraction (int, float)
- `*`: Multiplication (int, float, matrix multiplication)
- `/`: Division (int, float)
- `%`: Modulo (int only)
- `^`: Exponentiation (any numeric)

### Comparison Operators

- `==`: Equality
- `!=`: Inequality
- `<`, `>`: Less/greater than
- `<=`, `>=`: Less/greater than or equal

### Logical Operators

- `&&`: Logical AND
- `||`: Logical OR
- `!`: Logical NOT

### Bitwise Operators

- `&`: Bitwise AND
- `|`: Bitwise OR
- `^`: Bitwise XOR
- `~`: Bitwise NOT

### Assignment Operators

- `=`: Direct assignment
- `+=`: Add-assign
- `-=`: Subtract-assign
- `*=`: Multiply-assign
- `/=`: Divide-assign

### Special Operators

- `@`: Annotation operator
- `->`: Type annotation/lambda
- `=>`: Lambda arrow

## Control Flow

### If/Else

```
if (condition) statement
if (condition) statement else statement
```

### While Loop

```
while (condition) statement
```

### For Loop

```
for (init; condition; update) statement
```

### Return

```
return;
return expression;
```

## Functions

### Function Declaration

```
func add(a, b) {
  a + b
}

func addTyped(a: int, b: int) -> int {
  a + b
}
```

### Anonymous Functions

```
func(x) { x + 1 }
(x: int) -> int { x + 1 }
```

## Universes and Manifolds

### Universe Declaration

```
universe MyUniverse : "1.0.0" {
  // Universe contents
}
```

### Manifold Creation

Within a universe:

```
manifold space: euclidean(3) {
  // 3D Euclidean space
}

manifold spacetime: minkowski(4) {
  // 4D Minkowski spacetime
}

manifold curved: riemannian(3) {
  // 3D Riemannian manifold
}

manifold harmonic: pi-harmonic(5) {
  // 5D π-harmonic space
}
```

## Type Constraints

Apply constraints to types:

```
type PositiveInt = int { > 0 };
type SmallFloat = float { < 1.0, > 0.0 };
type NonEmptyString = string { length > 0 };
```

## Built-in Functions

### Mathematical Functions (40+)

#### Basic Operations
- `add(a, b)` → numeric
- `subtract(a, b)` → numeric
- `mul(a, b)` → numeric
- `div(a, b)` → numeric
- `mod(a: int, b: int)` → int
- `pow(a, b)` → float

#### Transcendental Functions
- `sqrt(x)` → float
- `abs(x)` → numeric
- `sin(x: float)` → float
- `cos(x: float)` → float
- `tan(x: float)` → float
- `log(x: float)` → float
- `exp(x: float)` → float
- `min(a, b)` → numeric
- `max(a, b)` → numeric

### Tensor Operations
- `transpose(m: matrix)` → matrix
- `inverse(m: matrix)` → matrix
- `det(m: matrix)` → float
- `trace(m: matrix)` → float
- `contract(t1: tensor, t2: tensor)` → tensor
- `expand(t: tensor)` → tensor
- `reshape(t: tensor, shape: tuple)` → tensor

### String Operations
- `concat(s1: string, s2: string)` → string
- `length(s: string)` → int
- `substr(s: string, start: int, end: int)` → string
- `upper(s: string)` → string
- `lower(s: string)` → string

### Collection Operations
- `length(x: any)` → int
- `push(x: any, item: any)` → void
- `pop(x: any)` → any
- `at(x: any, i: int)` → any
- `slice(x: any, start: int, end: int)` → any
- `map(x: any, f: func)` → any
- `filter(x: any, f: func)` → any
- `reduce(x: any, f: func, init: any)` → any

### Agent Operations
- `spawn(f: func)` → agent
- `join(a: agent)` → void
- `sync(a: agent)` → void
- `broadcast(a: agent, msg: any)` → void

### Universe Operations
- `universe_create(name: string)` → universe
- `universe_delete(name: string)` → void
- `universe_lookup(name: string)` → universe
- `universe_list()` → tuple

### Phase Operations
- `phase_push(s: phase_state)` → void
- `phase_pop()` → phase_state
- `phase_transition(s1: phase_state, s2: phase_state)` → phase_transition
- `phase_entangle(a1: agent, a2: agent)` → void

### I/O Operations
- `print(x: any)` → void
- `println(x: any)` → void
- `input()` → string

### Type Operations
- `type_of(x: any)` → string
- `is_type(x: any, t: string)` → bool
- `cast(x: any, t: string)` → any

## Examples

### Simple Arithmetic

```
let x = 5;
let y = 3;
let z = x + y;
println(z);  // Output: 8
```

### Type Inference

```
let a = 42;           // Inferred as int
let b = 3.14;         // Inferred as float
let c = "hello";      // Inferred as string
let d = true;         // Inferred as bool
```

### Functions

```
func factorial(n: int) -> int {
  if (n <= 1) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}
```

### Arrays and Objects

```
let arr = [1, 2, 3, 4, 5];
let person = { name: "Alice", age: 30 };
```

### Control Flow

```
for (let i = 0; i < 10; i += 1) {
  println(i);
}

while (true) {
  let input = input();
  if (input == "quit") {
    break;
  }
}
```

### Universe and Manifold

```
universe physics : "1.0.0" {
  manifold spacetime: minkowski(4) {
    // 4D spacetime
  }
}
```

### Tensor Operations

```
func matrixMultiply(a: matrix, b: matrix) -> matrix {
  // Matrix multiplication
  a * b
}
```

### Agent Coordination

```
func worker() {
  // Worker function
  println("Working...");
}

let agent = spawn(worker);
join(agent);  // Wait for completion
```

## Type Safety

K'UHUL enforces:

1. **Type Checking**: All expressions checked at parse time
2. **Type Inference**: Types inferred from context
3. **Constraints**: Custom constraints on types
4. **Semantic Validation**: Multi-pass analysis

## Semantic Analysis

The parser performs:

1. **Lexical Analysis**: Tokenization
2. **Syntax Analysis**: EBNF-compliant parsing
3. **Type Inference**: Algorithm for automatic type detection
4. **Semantic Analysis**: Scope checking, type validation
5. **Execution**: Eager evaluation by default

## Module System

Import external modules (future):

```
import math;
import geometry from "shapes";
```

Export definitions (future):

```
export func distance(p1: point, p2: point);
export type Vector3D;
```

## Error Handling

Error recovery and reporting:

```
try {
  // Code that might fail
} catch (e) {
  println("Error: " + e);
}
```

## Performance Characteristics

- **Parser**: O(n) single-pass
- **Type Checking**: O(n) with one semantic pass
- **Execution**: Eager evaluation (O(1) to O(n) depending on operation)
- **Tensor Operations**: Hardware-accelerated where possible

## Version Compatibility

Current: K'UHUL v7.0.0

Supports:
- ES6 modules
- Modern JavaScript (async/await)
- Node.js 14+

## Further Reading

- REPL Tutorial: See KUHUL_USER_GUIDE.md
- CLI Reference: See `kuhul help`
- Examples: See examples/ directory
