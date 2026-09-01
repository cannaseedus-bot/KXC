# ModelVersioning API Documentation

## Overview

The ModelVersioning module provides semantic versioning, rollback capabilities, and environment-based deployment for learned pattern models. It maintains a complete audit trail of all version changes and supports A/B testing across different environments.

## Installation

```javascript
import ModelVersioning from '../src/model-versioning.js';

const versioner = new ModelVersioning(dbPool, sqliteCache, {
  defaultEnvironment: 'dev',
  supportedEnvironments: ['dev', 'staging', 'production'],
  maxVersionHistory: 100,
  enableABTesting: true
});

await versioner.initialize();
```

## Class Constructor

```javascript
ModelVersioning(dbPool, sqliteCache, options = {})
```

### Parameters

- **dbPool** (DBPool): SQL Server connection pool. Use `null` to use default.
- **sqliteCache** (SQLiteCache): SQLite fallback cache. Use `null` to use default.
- **options** (Object):
  - `defaultEnvironment` (string): Default deployment environment. Default: `'dev'`
  - `supportedEnvironments` (Array): Valid environments. Default: `['dev', 'staging', 'production']`
  - `maxVersionHistory` (number): Max versions to keep. Default: `100`
  - `enableABTesting` (boolean): Support A/B testing. Default: `true`

## Methods

### initialize()

Initializes the versioning system and creates necessary database tables.

```javascript
const result = await versioner.initialize();
// Returns: { initialized: true }
```

### createVersion(patterns, metadata)

Creates a new model version with semantic versioning.

```javascript
const result = await versioner.createVersion(
  [
    { pattern_id: 'p1', confidence: 0.85, hit_count: 10 },
    { pattern_id: 'p2', confidence: 0.92, hit_count: 15 }
  ],
  {
    creator: 'ml-pipeline',
    change_summary: 'Improved accuracy on validation set'
  }
);

// Returns: { versionId: 'v1.0.1-1699123456789', versionNumber: '1.0.1' }
```

**Parameters:**
- `patterns` (Array): Array of pattern objects with metadata
- `metadata` (Object):
  - `creator` (string): User/system creating version
  - `change_summary` (string): Description of changes

**Returns:** Object with `versionId` and `versionNumber`

### getVersion(versionId)

Retrieves a specific version with all patterns.

```javascript
const version = await versioner.getVersion('v1.0.1-1699123456789');

// Returns version object with patterns:
// {
//   version_id: 'v1.0.1-1699123456789',
//   version_number: '1.0.1',
//   pattern_count: 2,
//   patterns: [...],
//   metadata: {...},
//   status: 'published',
//   created_at: 2024-01-01T12:00:00Z
// }
```

**Throws:**
- `Error` if version not found

### getCurrentVersion()

Gets the currently active (published) version.

```javascript
const current = await versioner.getCurrentVersion();

// Returns the latest published version object
```

**Returns:** Latest published version object

**Throws:**
- `Error` if no published version exists

### rollbackToVersion(versionId)

Rolls back to a previous version by creating a new version referencing it.

```javascript
const result = await versioner.rollbackToVersion('v1.0.0-1699123456780');

// Returns: {
//   success: true,
//   newVersionId: 'v1.0.2-1699123456900',
//   newVersionNumber: '1.0.2'
// }
```

**Note:** Rollback doesn't delete the current version; it creates a new version from the old one.

**Throws:**
- `Error` if version not found
- `Error` if target is already current

### listVersions(limit, offset)

Lists all versions with pagination.

```javascript
const versions = await versioner.listVersions(20, 0);

// Returns array of version objects, newest first
```

**Parameters:**
- `limit` (number): Max results per page. Default: `20`
- `offset` (number): Pagination offset. Default: `0`

### publishVersion(versionId, targetEnvironment)

Marks a version for deployment to an environment.

```javascript
const result = await versioner.publishVersion('v1.0.1-1699123456789', 'production');

// Returns: {
//   success: true,
//   versionId: 'v1.0.1-1699123456789',
//   environment: 'production'
// }
```

**Parameters:**
- `versionId` (string): Version to publish
- `targetEnvironment` (string): Target environment (dev, staging, production)

**Throws:**
- `Error` if environment not supported

### archiveVersion(versionId)

Archives an old version to manage storage.

```javascript
const result = await versioner.archiveVersion('v0.9.0-1699123456700');

// Returns: { success: true, versionId: '...', status: 'archived' }
```

### compareVersions(v1Id, v2Id)

Compares two versions and returns detailed differences.

```javascript
const comparison = await versioner.compareVersions(
  'v1.0.0-1699123456780',
  'v1.0.1-1699123456789'
);

// Returns: {
//   v1: { version_id: '...', version_number: '1.0.0' },
//   v2: { version_id: '...', version_number: '1.0.1' },
//   summary: {
//     added: 2,
//     removed: 1,
//     modified: 3,
//     unchanged: 10
//   },
//   details: {
//     added: [...],
//     removed: [...],
//     modified: [...]
//   }
// }
```

### getAuditTrail(limit)

Gets the audit trail of version changes.

```javascript
const trail = await versioner.getAuditTrail(50);

// Returns array of audit entries:
// [
//   {
//     action: 'create_version',
//     versionId: '...',
//     timestamp: '2024-01-01T12:00:00Z',
//     metadata: {...}
//   },
//   ...
// ]
```

**Actions tracked:**
- `create_version`: New version created
- `publish_version`: Version published to environment
- `rollback`: Rollback operation performed
- `archive_version`: Version archived

### getStats()

Gets current statistics.

```javascript
const stats = versioner.getStats();

// Returns: {
//   versionsCreated: 10,
//   versionsPublished: 8,
//   rollbackCount: 2,
//   abTestsActive: 1,
//   cachedVersions: 5,
//   auditTrailSize: 25
// }
```

## Database Schema

### model_versions table

```sql
CREATE TABLE [ModelVersions] (
  version_id NVARCHAR(128) PRIMARY KEY,
  version_number VARCHAR(20),
  pattern_count INT,
  metadata_json NVARCHAR(MAX),
  created_by VARCHAR(100),
  created_at DATETIME2,
  published_to_env VARCHAR(50),
  status VARCHAR(20),  -- draft, published, archived
  parent_version_id NVARCHAR(128),
  rollback_from NVARCHAR(128),
  change_summary NVARCHAR(MAX)
);
```

## Semantic Versioning

Versions follow MAJOR.MINOR.PATCH format:
- `MAJOR`: Breaking changes
- `MINOR`: New features
- `PATCH`: Bug fixes

Auto-increment strategy: Each new version increments PATCH automatically.

## Use Cases

### A/B Testing

```javascript
// Version 1 - Control
const v1 = await versioner.createVersion(patterns1);
await versioner.publishVersion(v1.versionId, 'staging');

// Version 2 - Variant
const v2 = await versioner.createVersion(patterns2);
await versioner.publishVersion(v2.versionId, 'dev');

// Route traffic based on environment
```

### Rolling Back on Issues

```javascript
// Current version has issues
const current = await versioner.getCurrentVersion();

// Rollback to previous
const previous = await versioner.listVersions(2, 1)[0];
const rollback = await versioner.rollbackToVersion(previous.version_id);

// New version (1.0.5) is created as a rollback
```

### Version Comparison

```javascript
// See what changed between versions
const comparison = await versioner.compareVersions(v1Id, v2Id);

if (comparison.summary.modified > 10) {
  console.log('Significant changes detected');
}
```

## Best Practices

1. **Always include metadata** when creating versions for traceability
2. **Use semantic versioning** conventions for clarity
3. **Archive old versions** to manage storage
4. **Monitor audit trail** for compliance and debugging
5. **Test versions in dev** before staging and production
6. **Enable A/B testing** when rolling out new patterns
7. **Set reasonable max history** to prevent unbounded growth

## Error Handling

```javascript
try {
  const version = await versioner.getVersion(versionId);
} catch (error) {
  if (error.message.includes('not found')) {
    console.error('Version does not exist');
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Integration with Other Modules

- **LearnedPatterns**: Provides patterns to version
- **SyncStrategy**: Syncs versions across instances
- **PatternDistribution**: Distributes versioned patterns
- **FederatedLearner**: Coordinates versioning across peers

## Performance Considerations

- Cache is limited to 100 versions in memory
- Old versions are automatically archived
- Audit trail is limited to 1000 entries
- Database indexes on status, created_at, environment

## Limits and Defaults

| Parameter | Default | Max |
|-----------|---------|-----|
| Max cached versions | N/A | 100 |
| Max version history | 100 | Unlimited |
| Max audit trail entries | 1000 | 10000 |
| Version number length | 20 chars | N/A |

## Event Logging

All operations log to console with format:
```
[ModelVersioning] ✓ Operation successful
[ModelVersioning] ✗ Operation failed: reason
[ModelVersioning] ⚠ Warning message
```
