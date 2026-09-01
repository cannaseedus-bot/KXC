# PatternDistribution API Documentation

## Overview

The PatternDistribution module enables cross-user pattern sharing with privacy-first anonymization, integrity verification through hashing, and distribution tracking. It supports multiple categories and distribution channels while maintaining user privacy.

## Installation

```javascript
import PatternDistribution from '../src/pattern-distribution.js';

const distributor = new PatternDistribution(dbPool, sqliteCache, {
  anonymizeData: true,
  categories: ['layer_selection', 'operation_dispatch', 'resource_optimization', 'gpu_decision'],
  distributionChannels: ['direct_sync', 'event_stream', 'queue'],
  privacyFirstApproach: true,
  maxCacheSize: 10000
});

await distributor.initialize();
```

## Class Constructor

```javascript
PatternDistribution(dbPool, sqliteCache, options = {})
```

### Parameters

- **dbPool** (DBPool): SQL Server connection pool. Use `null` for default.
- **sqliteCache** (SQLiteCache): SQLite cache. Use `null` for default.
- **options** (Object):
  - `anonymizeData` (boolean): Remove user identifiers. Default: `true`
  - `categories` (Array): Valid pattern categories
  - `distributionChannels` (Array): Available distribution methods
  - `privacyFirstApproach` (boolean): Enforce privacy. Default: `true`
  - `maxCacheSize` (number): Max patterns in memory. Default: `10000`

## Methods

### initialize()

Initializes the distribution system and creates tables.

```javascript
const result = await distributor.initialize();
// Returns: { initialized: true }
```

### publishPatterns(patterns, anonymized)

Publishes patterns for distribution (with optional anonymization).

```javascript
const result = await distributor.publishPatterns(
  [
    {
      pattern_id: 'p1',
      category: 'layer_selection',
      confidence: 0.85,
      hit_count: 100,
      user_id: 'user123'  // Will be removed if anonymized
    },
    {
      pattern_id: 'p2',
      category: 'gpu_decision',
      confidence: 0.92,
      hit_count: 50
    }
  ],
  true  // anonymize user data
);

// Returns: {
//   publishedCount: 2,
//   patternIds: ['pattern-1699123456789-abc123', 'pattern-1699123456790-def456']
// }
```

**Parameters:**
- `patterns` (Array): Pattern objects to publish
- `anonymized` (boolean): Remove user identifiers. Default: `true`

**Pattern Object Structure:**
```javascript
{
  pattern_id: string,           // Unique identifier
  category: string,             // Pattern category
  confidence: number,           // 0-1 confidence score
  hit_count: number,            // Usage count
  version: string,              // Version string
  pattern_data: object,         // Pattern details
  user_id: string,              // Will be anonymized if enabled
  node_id: string,              // Will be anonymized if enabled
  email: string                 // Will be anonymized if enabled
}
```

**Returns:** Object with `publishedCount` and array of `patternIds`

### subscribeToPatterns(subscriber)

Registers a subscriber for pattern updates.

```javascript
const result = await distributor.subscribeToPatterns({
  id: 'node-instance-1',
  categories: ['layer_selection', 'gpu_decision'],
  onUpdate: (event) => {
    console.log('Pattern update:', event);
  }
});

// Returns: {
//   subscribed: true,
//   subscriberId: 'node-instance-1'
// }
```

**Subscriber Object:**
```javascript
{
  id: string,              // Unique subscriber ID
  categories: Array,       // Pattern categories to receive
  onUpdate: Function,      // Callback for updates
  active: boolean          // Subscription active
}
```

**Event Format:**
```javascript
{
  event: 'pattern_published' | 'pattern_deprecated',
  patternId: string,
  category: string,
  timestamp: number
}
```

### getLatestPatterns(category, limit)

Retrieves the latest patterns, optionally filtered by category.

```javascript
const patterns = await distributor.getLatestPatterns('layer_selection', 10);

// Returns array of pattern objects with metadata:
// [
//   {
//     pattern_id: 'pattern-...',
//     category: 'layer_selection',
//     pattern_hash: 'sha256-hash...',
//     confidence: 0.85,
//     hit_count: 100,
//     version: '1.0.0',
//     published_at: 1699123456789,
//     distribution_count: 42,
//     anonymized: true
//   },
//   ...
// ]
```

**Parameters:**
- `category` (string): Optional category filter
- `limit` (number): Max results. Default: `10`

**Categories:**
- `layer_selection`: Layer choice patterns
- `operation_dispatch`: Operation routing patterns
- `resource_optimization`: Resource allocation patterns
- `gpu_decision`: GPU/CPU decision patterns

### syncPatternCache()

Synchronizes local cache with database.

```javascript
const result = await distributor.syncPatternCache();

// Returns: {
//   synced: 150,      // Total patterns in DB
//   updated: 5        // New patterns added to cache
// }
```

### getDistributionStats()

Gets distribution statistics.

```javascript
const stats = await distributor.getDistributionStats();

// Returns: {
//   patternsPublished: 125,
//   patternsDistributed: 3750,  // Total distributions
//   cacheHits: 2100,
//   cacheMisses: 50,
//   verificationFailures: 2,
//   deprecatedPatterns: 3,
//   dbStats: {
//     totalPatterns: 120,
//     totalDistributions: 3750,
//     uniqueCategories: 4,
//     avgConfidence: 0.82
//   },
//   cacheSize: 115,
//   subscriberCount: 8
// }
```

### deprecatePattern(patternId)

Marks a pattern as deprecated (obsolete).

```javascript
const result = await distributor.deprecatePattern('pattern-1699123456789-abc123');

// Returns: {
//   success: true,
//   patternId: 'pattern-1699123456789-abc123'
// }
```

### verifyPatternSignature(pattern)

Verifies pattern integrity using SHA256 hash.

```javascript
const isValid = await distributor.verifyPatternSignature({
  pattern_data: { layer: 'conv', kernel: 3 },
  pattern_hash: 'sha256-hash-value...'
});

// Returns: true | false
```

**Returns:** Boolean indicating signature validity

### recordDistribution(patternId, recipientId)

Records a pattern distribution to a recipient.

```javascript
const result = await distributor.recordDistribution(
  'pattern-1699123456789-abc123',
  'node-instance-2'
);

// Returns: {
//   success: true,
//   patternId: 'pattern-1699123456789-abc123',
//   recipientId: 'node-instance-2'
// }
```

### getPatternsForRecipient(recipientId, limit)

Gets patterns distributed to a specific recipient.

```javascript
const patterns = await distributor.getPatternsForRecipient('node-instance-2', 50);

// Returns array of pattern objects
```

## Database Schema

### DistributedPatterns table

```sql
CREATE TABLE [DistributedPatterns] (
  pattern_id NVARCHAR(128) PRIMARY KEY,
  category VARCHAR(50),
  pattern_hash VARCHAR(256),
  pattern_data NVARCHAR(MAX),
  confidence FLOAT,
  hit_count INT,
  version VARCHAR(20),
  published_at DATETIME2,
  distribution_count INT,
  anonymized BIT,
  deprecated_at DATETIME2
);
```

### PatternSubscribers table

```sql
CREATE TABLE [PatternSubscribers] (
  subscriber_id NVARCHAR(128) PRIMARY KEY,
  subscribed_at DATETIME2,
  categories NVARCHAR(MAX),
  active BIT,
  last_sync_at DATETIME2
);
```

## Anonymization

When `anonymizeData: true`, the following fields are removed:
- `user_id`
- `username`
- `email`
- `node_id`
- `instance_id`

**Before:**
```javascript
{
  pattern_id: 'p1',
  user_id: 'user123',
  email: 'user@example.com',
  confidence: 0.85
}
```

**After:**
```javascript
{
  pattern_id: 'p1',
  confidence: 0.85,
  category: 'layer_selection',
  version: '1.0.0'
}
```

## Hash Verification

Patterns include SHA256 hash for integrity:

```javascript
import crypto from 'crypto';

const pattern = { layer: 'conv', kernel: 3 };
const hash = crypto
  .createHash('sha256')
  .update(JSON.stringify(pattern))
  .digest('hex');

// Verify later
const isValid = hash === calculatedHash;
```

## Use Cases

### Publishing User Patterns

```javascript
const userPatterns = await extractUserPatterns(userId);

const result = await distributor.publishPatterns(userPatterns, true);
// User ID removed, patterns shared globally

console.log(`Published ${result.publishedCount} patterns`);
```

### Subscribing to Updates

```javascript
const subscription = await distributor.subscribeToPatterns({
  id: 'my-node',
  categories: ['gpu_decision'],
  onUpdate: (event) => {
    if (event.event === 'pattern_published') {
      console.log('New GPU decision pattern available');
      syncPatterns();
    }
  }
});
```

### Syncing Patterns

```javascript
const latest = await distributor.getLatestPatterns('layer_selection', 50);

for (const pattern of latest) {
  const isValid = await distributor.verifyPatternSignature(pattern);
  if (isValid) {
    await applyPattern(pattern);
    await distributor.recordDistribution(pattern.pattern_id, myNodeId);
  }
}
```

## Privacy Guarantees

1. **Data Removal**: User identifiers removed automatically
2. **Aggregation**: Patterns combined across users
3. **Hash Verification**: Integrity verification without plaintext
4. **Opt-in**: Users control pattern publishing
5. **Deprecation**: Old patterns can be removed

## Best Practices

1. **Always anonymize** sensitive user data
2. **Verify signatures** before applying patterns
3. **Record distributions** for tracking
4. **Subscribe with filters** for relevant patterns only
5. **Monitor stats** for distribution health
6. **Deprecate obsolete** patterns regularly
7. **Sync cache** periodically for freshness

## Error Handling

```javascript
try {
  const result = await distributor.publishPatterns(patterns);
} catch (error) {
  if (error.message.includes('Invalid category')) {
    console.error('Pattern category not supported');
  } else {
    console.error('Publication failed:', error);
  }
}
```

## Integration Points

- **ModelVersioning**: Distribute versioned patterns
- **FederatedLearner**: Share patterns across peers
- **SyncStrategy**: Sync distributions across instances
- **LearnedPatterns**: Source for publishable patterns

## Performance Notes

- Cache limited to 10,000 patterns
- Hash calculation on publish (one-time)
- Subscription callbacks are async
- Large pattern sets use pagination

## Audit Trail

All distributions are logged with:
- Pattern ID
- Recipient ID
- Timestamp
- Distribution count
- Verification status

## Compliance

- GDPR compliant (anonymization by default)
- No personal data in distribution channel
- Audit trail for data governance
- User control over pattern publishing
