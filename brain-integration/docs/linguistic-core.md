# Linguistic Core Engine - Phase 7

## Overview

The Linguistic Core Engine is a comprehensive text processing system for XJSON Phase 7. It provides sophisticated analysis through n-grams, bigrams, semantic coarse-grams, definition-grams, weights computation, and tensor operations.

**Key Capabilities:**
- N-gram extraction (2-5 gram lengths) with frequency analysis
- Bigram processing with PMI and cohesion metrics
- Semantic coarse-grain analysis with abstraction levels
- Definition extraction and component decomposition
- TF-IDF weight computation
- Advanced tensor operations and vector embeddings
- High-performance caching system
- Streaming and batch processing
- Semantic similarity computation

## Installation

```javascript
import { LinguisticCore } from './src/linguistic-core.js';
```

## Quick Start

```javascript
const core = new LinguisticCore();
core.initialize();

// Process text
const result = core.processText('the quick brown fox jumps over the lazy dog');

// Access processed data
console.log(result.tokens);           // Array of tokens
console.log(result.ngrams);           // Extracted n-grams
console.log(result.bigrams);          // Extracted bigrams
console.log(result.coarseGrams);      // Semantic groupings
console.log(result.weights);          // TF-IDF weights
console.log(result.tensorRepresentation); // Vector encoding
```

## API Reference

### LinguisticCore

#### Constructor

```javascript
const core = new LinguisticCore();
```

#### Methods

##### initialize(config)

Initialize the core with configuration options.

**Parameters:**
- `config` (Object, optional)
  - `cacheEnabled` (boolean): Enable result caching (default: true)
  - `maxCacheSize` (number): Maximum cached items (default: 1000)
  - `ngramLengths` (number[]): N-gram lengths to extract (default: [2,3,4,5])
  - `vectorDimension` (number): Vector embedding dimension (default: 768)
  - `semanticAbstractionLevels` (number): Abstraction levels (default: 5)
  - `tensorOptimization` (boolean): Enable tensor optimization (default: true)
  - `parallelProcessing` (boolean): Enable parallel processing (default: true)

**Example:**
```javascript
core.initialize({
  cacheEnabled: true,
  vectorDimension: 512,
  maxCacheSize: 2000
});
```

##### processText(text)

Main processing pipeline for text analysis.

**Parameters:**
- `text` (string): Text to process

**Returns:**
```javascript
{
  originalText: string,           // Original input text
  cleanText: string,              // Cleaned and normalized text
  tokens: string[],               // Tokenized words
  ngrams: Object[],               // N-gram analysis
  bigrams: Object[],              // Bigram analysis
  coarseGrams: Object[],          // Semantic groupings
  definitionGrams: Object[],      // Extracted definitions
  weights: Object[],              // TF-IDF weights
  tensorRepresentation: Object,   // Vector encoding
  metadata: {
    tokenCount: number,
    processedAt: string
  }
}
```

**Example:**
```javascript
const result = core.processText('Natural language processing is powerful');
console.log(result.tokens);     // ['natural', 'language', 'processing', ...]
console.log(result.ngrams);     // [{text: 'natural language', n: 2, ...}, ...]
```

##### getNGrams(text, n)

Extract n-grams of specified length.

**Parameters:**
- `text` (string): Input text
- `n` (number): N-gram length (default: 3, range: 2-10)

**Returns:** Array of n-gram objects with properties:
- `text` (string): The n-gram text
- `n` (number): N-gram length
- `frequency` (number): Occurrence count
- `positions` (number[]): Positions in text
- `weight` (number): Computed weight
- `overlap` (number): Overlap percentage

**Example:**
```javascript
const trigrams = core.getNGrams('the quick brown fox', 3);
// [{text: 'the quick brown', n: 3, frequency: 1, ...}]

const bigrams = core.getNGrams('the quick brown fox', 2);
// [{text: 'the quick', n: 2, ...}, {text: 'quick brown', ...}, ...]
```

##### getBigrams(text)

Extract bigrams (2-grams) with specialized metrics.

**Returns:** Array with additional properties:
- `word1` (string): First word
- `word2` (string): Second word
- `pmi` (number): Pointwise Mutual Information
- `cohesion` (number): Bigram cohesion score
- `mutualInformation` (number): Mutual information metric

**Example:**
```javascript
const bigrams = core.getBigrams('the dog and the cat');
// [{
//   text: 'the dog',
//   word1: 'the',
//   word2: 'dog',
//   pmi: 0.45,
//   cohesion: 0.75,
//   ...
// }]
```

##### getCoarseGrams(tokens)

Extract semantic coarse-grams from tokens.

**Parameters:**
- `tokens` (string[]): Tokenized text

**Returns:** Array of semantic groupings:
```javascript
{
  tokens: string[],                    // Grouped tokens
  semanticGroup: string,               // Category name
  abstractionLevel: number,            // 0-5 abstraction level
  similarity: number,                  // Group similarity score
  relationships: Map<string, number>,  // Token relationships
  contextWindow: string[],             // Context tokens
  frequency: number,                   // Occurrence count
  semanticStrength: number             // Overall group strength
}
```

**Example:**
```javascript
const tokens = ['happy', 'joyful', 'sad', 'depressed', 'dog', 'cat'];
const coarseGrams = core.getCoarseGrams(tokens);
// [{
//   tokens: ['happy', 'joyful'],
//   semanticGroup: 'emotions',
//   abstractionLevel: 2,
//   similarity: 0.85,
//   ...
// }]
```

##### getDefinitionGrams(tokens)

Extract definitions and components.

**Parameters:**
- `tokens` (string[]): Tokenized text

**Returns:** Array of definition objects:
```javascript
{
  term: string,                        // The defined term
  definition: string,                  // Definition text
  components: string[],                // Key components
  purpose: string,                     // Purpose/function
  relationships: Map<string, string>,  // Semantic relationships
  category: string,                    // Definition category
  confidence: number,                  // Confidence score
  startPosition: number,
  endPosition: number,
  patternType: string
}
```

**Example:**
```javascript
const tokens = ['a', 'dog', 'is', 'an', 'animal', 'that', 'barks'];
const defs = core.getDefinitionGrams(tokens);
```

##### computeWeights(grams)

Compute TF-IDF weights for n-grams.

**Parameters:**
- `grams` (Object[]): Array of gram objects

**Returns:** Weighted grams with properties:
- `weight` (number): TF-IDF weight
- `tf` (number): Term frequency
- `idf` (number): Inverse document frequency
- `normalized` (number): Normalized weight (0-1)

**Example:**
```javascript
const weights = core.computeWeights(result.ngrams);
// [{text: 'quick brown', weight: 0.456, tf: 0.5, idf: 0.912, ...}]
```

##### getTensorRepresentation(processedData)

Get vector tensor encoding of text.

**Returns:**
```javascript
{
  vector: Float32Array,              // 768-dimensional vector
  shape: number[],                   // [768]
  dtype: string,                     // 'float32'
  metadata: {
    ngramCount: number,
    encodedAt: string,
    dimension: number
  }
}
```

**Example:**
```javascript
const tensor = core.getTensorRepresentation(result);
console.log(tensor.vector.length);  // 768
```

##### getSemanticSimilarity(data1, data2)

Compute cosine similarity between two texts.

**Parameters:**
- `data1` (Object): Processed data from processText
- `data2` (Object): Processed data from processText

**Returns:** Similarity score (0-1)

**Example:**
```javascript
const result1 = core.processText('the cat is a pet');
const result2 = core.processText('the dog is a pet');
const similarity = core.getSemanticSimilarity(result1, result2);
console.log(similarity);  // ~0.85
```

##### getKeyTerms(processedData, topN)

Extract key terms by importance.

**Parameters:**
- `processedData` (Object): Result from processText
- `topN` (number): Number of terms to return (default: 10)

**Returns:** Array of terms:
```javascript
[
  {
    term: string,        // The term text
    score: number,       // Importance score
    frequency: number,   // Occurrence count
    type: string         // 'ngram'
  }
]
```

**Example:**
```javascript
const terms = core.getKeyTerms(result, 5);
// [{term: 'brown fox', score: 0.87, frequency: 1, type: 'ngram'}]
```

##### batchProcess(texts)

Process multiple texts efficiently.

**Parameters:**
- `texts` (string[]): Array of texts

**Returns:** Array of results:
```javascript
[
  { success: true, data: Object, index: number },
  { success: false, error: string, index: number }
]
```

**Example:**
```javascript
const results = core.batchProcess([
  'first text',
  'second text',
  'third text'
]);
results.forEach(r => {
  if (r.success) console.log(r.data);
  else console.log(r.error);
});
```

##### streamProcess(text, chunkSize)

Stream process large texts.

**Parameters:**
- `text` (string): Large text
- `chunkSize` (number): Sentences per chunk (default: 1000)

**Yields:** Processing results for each chunk

**Example:**
```javascript
for (const chunk of core.streamProcess(largeText, 500)) {
  console.log(chunk.tokens.length);
}
```

##### getMetrics()

Get performance statistics.

**Returns:**
```javascript
{
  totalProcessed: number,
  totalTime: string,
  averageTime: string,
  cacheHits: number,
  cacheMisses: number,
  cacheHitRate: string,
  cacheSize: number,
  maxCacheSize: number
}
```

**Example:**
```javascript
const metrics = core.getMetrics();
console.log(metrics.cacheHitRate);      // "45.00%"
console.log(metrics.averageTime);       // "2.34"
```

##### clearCache()

Clear the result cache.

```javascript
core.clearCache();
```

##### reset()

Reset all statistics and cache.

```javascript
core.reset();
```

## Data Structures

### N-Gram Object

```javascript
{
  text: string,                    // The n-gram text
  n: number,                       // Length (2-10)
  frequency: number,               // Occurrence count
  positions: number[],             // Positions in text
  weight: number,                  // Computed weight
  embedding: Float32Array,         // 768-dim vector
  overlap: number,                 // Overlap percentage
  entropy: number                  // Information entropy
}
```

### Bigram Object (extends N-Gram)

```javascript
{
  text: string,
  n: 2,
  word1: string,                   // First word
  word2: string,                   // Second word
  frequency: number,
  positions: number[],
  weight: number,
  embedding: Float32Array,
  pmi: number,                     // Pointwise Mutual Info
  cohesion: number,                // Cohesion score
  mutualInformation: number        // MI metric
}
```

### Semantic Coarse-Gram Object

```javascript
{
  tokens: string[],                // Grouped tokens
  semanticGroup: string,           // Category
  abstractionLevel: number,        // 0-5
  similarity: number,              // Group similarity
  relationships: Map,              // Token relationships
  positions: number[],             // Positions in text
  contextWindow: string[],         // Context tokens
  frequency: number,               // Group frequency
  semanticStrength: number         // Overall strength
}
```

### Definition Gram Object

```javascript
{
  term: string,                    // Defined term
  definition: string,              // Definition text
  components: string[],            // Key parts
  purpose: string,                 // Function/purpose
  relationships: Map,              // Semantic links
  category: string,                // Definition type
  confidence: number,              // Confidence 0-1
  startPosition: number,
  endPosition: number,
  patternType: string              // Pattern matched
}
```

### Tensor Object

```javascript
{
  vector: Float32Array,            // Data array
  shape: number[],                 // [768] for embeddings
  dtype: string,                   // 'float32'
  metadata: {
    ngramCount: number,
    encodedAt: string,
    dimension: number
  }
}
```

## Configuration Options

### Default Configuration

```javascript
{
  cacheEnabled: true,
  maxCacheSize: 1000,
  ngramLengths: [2, 3, 4, 5],
  vectorDimension: 768,
  semanticAbstractionLevels: 5,
  tensorOptimization: true,
  parallelProcessing: true,
  vocabularySize: 50000,
  minFrequency: 1,
  similarityThreshold: 0.6,
  maxGroupSize: 10
}
```

### Performance Tuning

```javascript
// For speed (reduced accuracy)
core.initialize({
  cacheEnabled: true,
  maxCacheSize: 2000,
  vectorDimension: 256,
  tensorOptimization: true
});

// For accuracy (reduced speed)
core.initialize({
  cacheEnabled: true,
  maxCacheSize: 5000,
  vectorDimension: 768,
  semanticAbstractionLevels: 8,
  parallelProcessing: true
});

// For memory conservation
core.initialize({
  cacheEnabled: false,
  vectorDimension: 128,
  maxCacheSize: 100
});
```

## Usage Examples

### Basic Text Analysis

```javascript
const core = new LinguisticCore();
core.initialize();

const text = 'The quick brown fox jumps over the lazy dog';
const result = core.processText(text);

console.log('Tokens:', result.tokens);
console.log('Key terms:', core.getKeyTerms(result, 3));
console.log('Metrics:', core.getMetrics());
```

### Semantic Analysis

```javascript
const core = new LinguisticCore();
core.initialize();

const text = 'happy dogs and joyful cats';
const result = core.processText(text);

// Get semantic groupings
console.log('Semantic groups:', result.coarseGrams);

// Extract definitions
console.log('Definitions:', result.definitionGrams);

// Analyze abstraction
const hierarchy = core.semanticCoarseGrammer
  .extractAbstractionHierarchy(result.tokens);
console.log('Concrete terms:', hierarchy.concrete);
console.log('Abstract terms:', hierarchy.abstract);
```

### Similarity Analysis

```javascript
const core = new LinguisticCore();
core.initialize();

const text1 = 'The cat is a furry pet';
const text2 = 'The dog is a furry animal';
const text3 = 'Programming is fun';

const result1 = core.processText(text1);
const result2 = core.processText(text2);
const result3 = core.processText(text3);

const sim12 = core.getSemanticSimilarity(result1, result2);
const sim13 = core.getSemanticSimilarity(result1, result3);

console.log('Similarity (cat vs dog):', sim12);  // Higher
console.log('Similarity (cat vs prog):', sim13); // Lower
```

### Batch Processing

```javascript
const core = new LinguisticCore();
core.initialize({ cacheEnabled: true });

const documents = [
  'Document about machine learning',
  'Document about neural networks',
  'Document about deep learning',
  'Document about cooking recipes'
];

const results = core.batchProcess(documents);

results.forEach((r, i) => {
  if (r.success) {
    const terms = core.getKeyTerms(r.data, 3);
    console.log(`Doc ${i}:`, terms.map(t => t.term).join(', '));
  }
});

console.log('Cache hit rate:', core.getMetrics().cacheHitRate);
```

### Stream Processing Large Files

```javascript
const core = new LinguisticCore();
core.initialize();

const largeText = fs.readFileSync('book.txt', 'utf-8');

for (const chunk of core.streamProcess(largeText, 500)) {
  console.log(`Processed ${chunk.tokenCount} tokens`);
  const terms = core.getKeyTerms(chunk, 5);
  console.log('Key terms:', terms.map(t => t.term));
}
```

### Advanced Tensor Operations

```javascript
const core = new LinguisticCore();
core.initialize();

const result1 = core.processText('information technology');
const result2 = core.processText('computer science');

const tensor1 = result1.tensorRepresentation;
const tensor2 = result2.tensorRepresentation;

// Compute similarity
const similarity = core.getTensorEngine()
  .computeCosineSimilarity(tensor1.vector, tensor2.vector);

// Add tensors
const engine = core.getTensorEngine();
const combined = engine.tensorAdd(
  { data: tensor1.vector, shape: [768] },
  { data: tensor2.vector, shape: [768] }
);

// Scale tensor
const scaled = engine.tensorScalarMultiply(
  { data: tensor1.vector, shape: [768] },
  0.5
);
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| processText | O(n log n) | n = text length |
| getNGrams | O(n) | Linear in text |
| computeWeights | O(n²) | n = gram count |
| tensorOps | O(d) | d = dimension (768) |
| getSemanticSimilarity | O(d) | Vector dimension |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Cache | O(c*n) | c = cache size, n = text size |
| Embeddings | O(d) | d = 768 dimensions |
| Tensor | O(d) | Vector storage |
| N-grams | O(n) | Linear with unique grams |

### Benchmarks (on typical hardware)

```
Text size: 1000 tokens
- processText: ~5-10ms (cached)
- getNGrams: ~2-4ms
- computeWeights: ~5-8ms
- tensorRepresentation: ~3-5ms
- getSemanticSimilarity: ~1-2ms

Caching Impact:
- First pass: ~25ms
- Cached pass: ~1-2ms
- Cache overhead: <100KB per entry
```

## Integration

### With MX2LM CLI

```javascript
import { LinguisticCore } from './src/linguistic-core.js';

export function createLinguisticAnalyzer() {
  const core = new LinguisticCore();
  core.initialize({ cacheEnabled: true });
  return core;
}
```

### With GUIX Model

```javascript
const core = new LinguisticCore();
core.initialize();

// Feed processed linguistic data to GUIX
const linguistic = core.processText(inputText);
const embedding = linguistic.tensorRepresentation.vector;
const terms = core.getKeyTerms(linguistic, 20);

guixModel.processLinguisticData({
  embedding,
  keyTerms: terms,
  ngrams: linguistic.ngrams
});
```

### With Kuhul CLI

```javascript
const core = new LinguisticCore();
core.initialize({ parallelProcessing: true });

// Batch analyze documents
const results = core.batchProcess(documents);

// Export results for Kuhul processing
export const linguisticAnalysis = {
  results,
  metrics: core.getMetrics()
};
```

## Error Handling

```javascript
const core = new LinguisticCore();

try {
  core.initialize(invalidConfig);
} catch (error) {
  console.error('Initialization failed:', error.message);
}

try {
  core.processText('');  // Empty text
} catch (error) {
  console.error('Processing failed:', error.message);
}

// Batch with error handling
const results = core.batchProcess(['valid text', 123, 'another valid']);
results.forEach((r, i) => {
  if (r.success) {
    console.log(`Item ${i}: Success`);
  } else {
    console.error(`Item ${i}: ${r.error}`);
  }
});
```

## Testing

Run the comprehensive test suite:

```bash
node tests/test-linguistic-core.mjs
```

**Test Coverage:**
- 40+ test cases
- N-gram extraction and analysis
- Bigram processing with metrics
- Semantic coarse-gram extraction
- Definition gram extraction
- Weight computation
- Tensor operations
- Similarity metrics
- Integration tests
- Performance benchmarks

## Troubleshooting

### High Memory Usage
- Reduce `maxCacheSize`
- Disable caching with `cacheEnabled: false`
- Reduce `vectorDimension`
- Use `streamProcess` for large texts

### Slow Processing
- Enable `tensorOptimization`
- Increase `maxCacheSize` for repeated texts
- Enable `parallelProcessing`
- Pre-initialize with common texts

### Inconsistent Results
- Check random seed consistency
- Verify tokenization parameters
- Validate input text encoding (UTF-8)

## Standards Compliance

- **ES6 Modules**: Fully ESM compatible
- **Zero Dependencies**: No external packages required
- **Node.js**: 14.0+
- **Browser**: Modern browsers with ES6 support
- **Production Ready**: Error handling, validation, logging

## Version

Version: 1.0.0
Phase: 7
Release Date: 2024
