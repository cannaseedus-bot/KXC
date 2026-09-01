# Linguistic Core Engine - Integration Guide

## Quick Integration Steps

### 1. Basic Import
```javascript
import { LinguisticCore } from './src/linguistic-core.js';
```

### 2. Initialize
```javascript
const core = new LinguisticCore();
core.initialize({
  cacheEnabled: true,
  vectorDimension: 768,
  maxCacheSize: 1000
});
```

### 3. Process Text
```javascript
const result = core.processText('Your text here');
```

### 4. Use Results
```javascript
// Get tokens
const tokens = result.tokens;

// Get n-grams
const ngrams = result.ngrams;

// Get key terms
const keyTerms = core.getKeyTerms(result, 10);

// Get tensor representation
const vector = result.tensorRepresentation.vector;

// Get semantic similarity
const similarity = core.getSemanticSimilarity(result1, result2);
```

---

## Integration with MX2LM CLI

```javascript
import { LinguisticCore } from './src/linguistic-core.js';

export class MX2LMTextProcessor {
  constructor() {
    this.linguistic = new LinguisticCore();
    this.linguistic.initialize({
      cacheEnabled: true,
      ngramLengths: [2, 3, 4, 5]
    });
  }

  processInput(text) {
    const result = this.linguistic.processText(text);
    return {
      tokens: result.tokens,
      features: {
        ngrams: result.ngrams.slice(0, 20),
        bigrams: result.bigrams.slice(0, 20),
        semanticGroups: result.coarseGrams,
        definitions: result.definitionGrams
      },
      embedding: result.tensorRepresentation.vector,
      keyTerms: this.linguistic.getKeyTerms(result, 10)
    };
  }

  compareTexts(text1, text2) {
    const r1 = this.linguistic.processText(text1);
    const r2 = this.linguistic.processText(text2);
    return {
      similarity: this.linguistic.getSemanticSimilarity(r1, r2),
      commonTerms: this.findCommonTerms(r1, r2)
    };
  }

  findCommonTerms(r1, r2) {
    const t1 = new Set(r1.tokens);
    const t2 = new Set(r2.tokens);
    return [...t1].filter(t => t2.has(t));
  }
}
```

---

## Integration with GUIX Model

```javascript
import { LinguisticCore } from './src/linguistic-core.js';

export class GUIXLinguisticFeatureExtractor {
  constructor() {
    this.linguistic = new LinguisticCore();
    this.linguistic.initialize({ vectorDimension: 768 });
  }

  extractFeatures(text) {
    const result = this.linguistic.processText(text);

    return {
      // Embeddings for neural processing
      embedding: result.tensorRepresentation.vector,
      
      // N-gram features
      ngramFeatures: result.ngrams.map(ng => ({
        text: ng.text,
        weight: ng.weight,
        frequency: ng.frequency
      })),

      // Semantic features
      semanticFeatures: result.coarseGrams.map(cg => ({
        tokens: cg.tokens,
        group: cg.semanticGroup,
        strength: cg.semanticStrength
      })),

      // Definition features
      definitionFeatures: result.definitionGrams.map(dg => ({
        term: dg.term,
        category: dg.category,
        components: dg.components
      })),

      // Metadata
      metadata: {
        tokenCount: result.tokens.length,
        ngramCount: result.ngrams.length,
        semanticGroupCount: result.coarseGrams.length,
        definitionCount: result.definitionGrams.length
      }
    };
  }

  batchExtractFeatures(texts) {
    return texts.map(text => this.extractFeatures(text));
  }

  computeCorpusStatistics(texts) {
    const results = texts.map(t => this.linguistic.processText(t));
    
    const allNgrams = [];
    const allTokens = [];
    
    results.forEach(r => {
      allNgrams.push(...r.ngrams);
      allTokens.push(...r.tokens);
    });

    return {
      totalTexts: texts.length,
      totalTokens: allTokens.length,
      uniqueTokens: new Set(allTokens).size,
      totalNgrams: allNgrams.length,
      uniqueNgrams: new Set(allNgrams.map(n => n.text)).size,
      averageTextLength: allTokens.length / texts.length
    };
  }
}
```

---

## Integration with Kuhul CLI

```javascript
import { LinguisticCore } from './src/linguistic-core.js';

export class KuhulDocumentAnalyzer {
  constructor(configPath) {
    this.linguistic = new LinguisticCore();
    this.linguistic.initialize({
      cacheEnabled: true,
      maxCacheSize: 5000,
      parallelProcessing: true
    });
    this.documents = [];
  }

  analyzeDocument(filepath, content) {
    const result = this.linguistic.processText(content);
    
    const analysis = {
      filepath,
      statistics: {
        tokenCount: result.tokens.length,
        ngramCount: result.ngrams.length,
        uniqueNgrams: new Set(result.ngrams.map(n => n.text)).size,
        semanticGroupCount: result.coarseGrams.length,
        definitionCount: result.definitionGrams.length
      },
      keyTerms: this.linguistic.getKeyTerms(result, 20),
      semanticProfile: result.coarseGrams,
      embedding: result.tensorRepresentation.vector
    };

    this.documents.push(analysis);
    return analysis;
  }

  analyzeDirectory(dirPath) {
    const results = [];
    // Pseudo code - actual implementation depends on file system
    // const files = fs.readdirSync(dirPath);
    // files.forEach(f => results.push(this.analyzeDocument(f, content)));
    return results;
  }

  findSimilarDocuments(targetIndex, topN = 10) {
    const target = this.documents[targetIndex];
    const similarities = [];

    for (let i = 0; i < this.documents.length; i++) {
      if (i === targetIndex) continue;

      const doc1 = this.linguistic.processText(/* content1 */);
      const doc2 = this.linguistic.processText(/* content2 */);
      
      const similarity = this.linguistic.getSemanticSimilarity(doc1, doc2);
      similarities.push({
        docIndex: i,
        filepath: this.documents[i].filepath,
        similarity
      });
    }

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);
  }

  generateReport() {
    const metrics = this.linguistic.getMetrics();
    
    return {
      documentsAnalyzed: this.documents.length,
      totalProcessingTime: metrics.totalTime,
      averageTimePerDocument: metrics.averageTime,
      cachePerformance: {
        hits: metrics.cacheHits,
        misses: metrics.cacheMisses,
        hitRate: metrics.cacheHitRate
      },
      topTerms: this.documents
        .flatMap(d => d.keyTerms)
        .sort((a, b) => b.score - a.score)
        .slice(0, 50)
    };
  }
}
```

---

## Advanced Usage Patterns

### Pattern 1: Semantic Clustering
```javascript
const core = new LinguisticCore();
core.initialize();

const documents = [...];
const clusters = new Map();

for (const doc of documents) {
  const result = core.processText(doc);
  const mainGroup = result.coarseGrams[0]?.semanticGroup;
  
  if (!clusters.has(mainGroup)) {
    clusters.set(mainGroup, []);
  }
  clusters.get(mainGroup).push({
    doc,
    embedding: result.tensorRepresentation.vector
  });
}

// Clusters now contain semantically grouped documents
```

### Pattern 2: Batch Similarity Analysis
```javascript
const core = new LinguisticCore();
core.initialize({ cacheEnabled: true });

const texts = [...];
const results = texts.map(t => core.processText(t));

const similarityMatrix = [];
for (let i = 0; i < results.length; i++) {
  const row = [];
  for (let j = 0; j < results.length; j++) {
    const sim = i === j ? 1 : 
      core.getSemanticSimilarity(results[i], results[j]);
    row.push(sim);
  }
  similarityMatrix.push(row);
}

// Use similarity matrix for clustering, ranking, etc.
```

### Pattern 3: Definition-based Categorization
```javascript
const core = new LinguisticCore();
core.initialize();

const documents = [...];
const categories = new Map();

for (const doc of documents) {
  const result = core.processText(doc);
  
  result.definitionGrams.forEach(def => {
    const cat = def.category;
    if (!categories.has(cat)) {
      categories.set(cat, []);
    }
    categories.get(cat).push({
      doc,
      definition: def.definition,
      term: def.term
    });
  });
}

// Automatically categorize documents by definitions found
```

### Pattern 4: Stream Processing Large Corpus
```javascript
const core = new LinguisticCore();
core.initialize({ cacheEnabled: true });

async function processLargeFile(filepath) {
  const content = await fs.promises.readFile(filepath, 'utf-8');
  
  for (const chunk of core.streamProcess(content, 1000)) {
    // Process each chunk
    const keyTerms = core.getKeyTerms(chunk, 10);
    console.log('Key terms:', keyTerms);
    
    // Can do database inserts, API calls, etc per chunk
    // without holding entire file in processed memory
  }
}
```

---

## Performance Optimization Tips

### For High-Frequency Processing
```javascript
// Enable aggressive caching
core.initialize({
  cacheEnabled: true,
  maxCacheSize: 10000,  // Larger cache
  vectorDimension: 512   // Faster embedding
});
```

### For Large-Scale Analysis
```javascript
// Use stream processing
for (const chunk of core.streamProcess(text, 5000)) {
  // Process incrementally
}

// Or batch without holding all in memory
const batchSize = 100;
for (let i = 0; i < docs.length; i += batchSize) {
  const batch = docs.slice(i, i + batchSize);
  core.batchProcess(batch);
}
```

### For Memory-Constrained Environments
```javascript
core.initialize({
  cacheEnabled: false,      // No caching
  maxCacheSize: 100,        // Minimal cache
  vectorDimension: 128,     // Smaller embeddings
  tensorOptimization: false  // Simpler operations
});
```

---

## Troubleshooting Integration

### Issue: Slow Processing
**Solution:** Enable caching and increase maxCacheSize
```javascript
core.initialize({ cacheEnabled: true, maxCacheSize: 5000 });
```

### Issue: High Memory Usage
**Solution:** Disable caching or reduce vector dimension
```javascript
core.initialize({ cacheEnabled: false, vectorDimension: 256 });
```

### Issue: Inconsistent Results
**Solution:** Embeddings are deterministic, ensure consistent initialization
```javascript
// Same text will always produce same embedding
const r1 = core.processText('hello');
const r2 = core.processText('hello');
// r1.tensorRepresentation.vector === r2.tensorRepresentation.vector ✓
```

---

## API Reference Quick Lookup

| Method | Purpose | Returns |
|--------|---------|---------|
| `processText(text)` | Main pipeline | Full analysis object |
| `getNGrams(text, n)` | Extract n-grams | Array of n-gram objects |
| `getBigrams(text)` | Extract bigrams | Array of bigram objects |
| `getCoarseGrams(tokens)` | Semantic grouping | Array of semantic groups |
| `getDefinitionGrams(tokens)` | Extract definitions | Array of definitions |
| `computeWeights(grams)` | TF-IDF weighting | Weighted grams |
| `getTensorRepresentation(data)` | Vector encoding | Tensor object |
| `getSemanticSimilarity(d1, d2)` | Compare texts | Similarity score 0-1 |
| `getKeyTerms(data, topN)` | Top terms | Array of term objects |
| `batchProcess(texts)` | Process multiple | Results array |
| `streamProcess(text, size)` | Large text streaming | Generator of chunks |
| `getMetrics()` | Performance stats | Metrics object |
| `clearCache()` | Clear cache | void |
| `reset()` | Reset statistics | void |

---

## Version & Support

- **Version:** 1.0.0
- **Phase:** 7
- **Node.js:** 14.0+
- **Dependencies:** None
- **Status:** Production Ready

For detailed documentation, see:
- `brain-integration/docs/linguistic-core.md` - Complete API reference
- `brain-integration/docs/LINGUISTIC_CORE_IMPLEMENTATION.md` - Implementation details
