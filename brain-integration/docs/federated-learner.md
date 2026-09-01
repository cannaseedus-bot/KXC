# FederatedLearner API Documentation

## Overview

The FederatedLearner module enables peer-to-peer pattern coordination with local-first learning, consensus building through voting, and graceful degradation when networks partition. It supports both synchronous and asynchronous pattern agreement.

## Installation

```javascript
import FederatedLearner from '../src/federated-learner.js';

const learner = new FederatedLearner(dbPool, sqliteCache, {
  localNodeId: 'node-prod-1',
  peerNodes: [
    { id: 'node-prod-2' },
    { id: 'node-prod-3' }
  ],
  consensusThreshold: 0.66,
  minPeersForConsensus: 2,
  aggregationStrategy: 'weighted_average',
  syncIntervalMs: 30000
});

await learner.initialize();
```

## Class Constructor

```javascript
FederatedLearner(dbPool, sqliteCache, options = {})
```

### Parameters

- **dbPool** (DBPool): SQL Server connection pool
- **sqliteCache** (SQLiteCache): SQLite fallback cache
- **options** (Object):
  - `localNodeId` (string): Unique node identifier
  - `peerNodes` (Array): Array of peer node objects with `id`
  - `consensusThreshold` (number): Required vote ratio (0-1). Default: `0.66`
  - `minPeersForConsensus` (number): Min peers for consensus. Default: `2`
  - `aggregationStrategy` (string): Aggregation method. Default: `'weighted_average'`
  - `syncIntervalMs` (number): Background sync interval. Default: `30000`
  - `enableLocalLearning` (boolean): Allow local patterns. Default: `true`
  - `maxProposalAge` (number): Proposal TTL in ms. Default: `3600000` (1 hour)

## Methods

### initialize()

Initializes the federated learner.

```javascript
const result = await learner.initialize();
// Returns: { initialized: true }
```

### addLocalPattern(pattern, confidence)

Adds a pattern learned locally on this node.

```javascript
const result = await learner.addLocalPattern(
  {
    layer: 'conv2d',
    kernel_size: 3,
    activation: 'relu'
  },
  0.85  // confidence score
);

// Returns: {
//   patternId: 'pattern-1699123456789-abc123',
//   stored: true
// }
```

**Parameters:**
- `pattern` (Object): Pattern data to store
- `confidence` (number): Confidence in pattern (0-1). Default: `0.5`

**Returns:** Object with `patternId` and `stored` flag

### proposePeerUpdate(pattern)

Proposes a pattern to peers for adoption via voting.

```javascript
const result = await learner.proposePeerUpdate({
  pattern_id: 'p1',
  layer: 'conv2d',
  kernel_size: 5,
  confidence: 0.89
});

// Returns: {
//   proposalId: 'proposal-1699123456789-def456',
//   status: 'pending'
// }
```

**Note:** Proposer automatically votes to accept

**Returns:** Object with `proposalId` and initial `status`

### voteOnProposal(proposalId, vote, confidence)

Vote on a peer's proposed pattern.

```javascript
const result = await learner.voteOnProposal(
  'proposal-1699123456789-def456',
  'accept',  // or 'reject'
  0.88       // confidence in vote
);

// Returns: {
//   proposalId: 'proposal-1699123456789-def456',
//   vote: 'accept',
//   consensusReached: true | false
// }
```

**Parameters:**
- `proposalId` (string): Proposal to vote on
- `vote` (string): `'accept'` or `'reject'`
- `confidence` (number): Vote confidence (0-1). Default: `0.5`

**Returns:** Vote result with consensus status

**Throws:**
- `Error` if proposal not found
- `Error` if invalid vote value

### aggregatePatterns(patterns)

Combines patterns from multiple sources using aggregation strategy.

```javascript
const result = await learner.aggregatePatterns([
  { pattern_id: 'p1', confidence: 0.85, hit_count: 100 },
  { pattern_id: 'p2', confidence: 0.92, hit_count: 150 },
  { pattern_id: 'p3', confidence: 0.78, hit_count: 75 }
]);

// Returns: {
//   aggregation_id: 'agg-1699123456789-...',
//   confidence: 0.851,          // Aggregated
//   hit_count: 108,             // Average
//   aggregation_strategy: 'weighted_average',
//   source_patterns: 3,
//   pattern_data: { /* merged */ }
// }
```

**Aggregation Strategies:**
- `weighted_average`: Confidence-weighted average
- `simple_average`: (sum of confidences) / count
- `max_confidence`: Highest confidence value

**Parameters:**
- `patterns` (Array): Array of pattern objects. Must not be empty.

**Returns:** Aggregated pattern object

**Throws:**
- `Error` if empty array
- `Error` if not an array

### getConsensus(patternId, threshold)

Gets consensus status for a pattern proposal.

```javascript
const consensus = await learner.getConsensus(
  'p1',
  0.75  // Optional custom threshold
);

// Returns: {
//   patternId: 'p1',
//   hasConsensus: true,
//   consensusRatio: 0.85,      // 85% accept votes
//   voteCount: 7,              // Total votes
//   acceptVotes: 6,
//   rejectVotes: 1,
//   threshold: 0.75
// }
```

**Parameters:**
- `patternId` (string): Pattern to check consensus for
- `threshold` (number): Optional override consensus threshold

**Returns:** Consensus status object

### syncWithPeers()

Synchronizes patterns with peer nodes.

```javascript
const result = await learner.syncWithPeers();

// Returns: {
//   synced: 3,                 // Successful syncs
//   failed: 0,                 // Failed syncs
//   networkPartitioned: false  // Network health
// }
```

**Automatic Actions:**
- Exchange local patterns with peers
- Collect votes on proposals
- Detect network partitions
- Update peer status

**Returns:** Sync result object

**Note:** Called automatically via background worker if configured

### getPeerStatus()

Gets peer network topology status.

```javascript
const status = await learner.getPeerStatus();

// Returns: {
//   localNodeId: 'node-prod-1',
//   totalPeers: 3,
//   onlinePeers: 2,
//   networkPartitioned: false,
//   lastSyncTime: 1699123456789,
//   peers: {
//     'node-prod-2': {
//       peer_id: 'node-prod-2',
//       isOnline: true,
//       lastSeen: 5234,         // ms ago
//       patterns_synced: 10
//     },
//     'node-prod-3': {
//       peer_id: 'node-prod-3',
//       isOnline: false,
//       lastSeen: 120000
//     }
//   }
// }
```

**Returns:** Detailed peer network status

### getStats()

Gets federated learning statistics.

```javascript
const stats = learner.getStats();

// Returns: {
//   patternsAdded: 25,
//   proposalsMade: 8,
//   proposalsAccepted: 6,
//   proposalsRejected: 2,
//   consensusesReached: 6,
//   syncAttempts: 24,
//   successfulSyncs: 22,
//   failedSyncs: 2,
//   averageConsensusTime: 1250,  // ms
//   activePatterns: 25,
//   activeProposals: 0,
//   networkPartitioned: false,
//   onlinePeers: 2
// }
```

## Database Schema

### LocalPatterns table

```sql
CREATE TABLE [LocalPatterns] (
  pattern_id NVARCHAR(128) PRIMARY KEY,
  pattern_data NVARCHAR(MAX),
  confidence FLOAT,
  learned_by VARCHAR(100),
  learned_at DATETIME2,
  hit_count INT,
  version VARCHAR(20)
);
```

### FederatedProposals table

```sql
CREATE TABLE [FederatedProposals] (
  proposal_id NVARCHAR(256) PRIMARY KEY,
  pattern_id NVARCHAR(128),
  proposed_by VARCHAR(100),
  consensus_threshold FLOAT,
  aggregated_confidence FLOAT,
  status VARCHAR(50),  -- pending, accepted, rejected
  created_at DATETIME2,
  resolved_at DATETIME2,
  pattern_data NVARCHAR(MAX)
);
```

### ProposalVotes table

```sql
CREATE TABLE [ProposalVotes] (
  vote_id NVARCHAR(256) PRIMARY KEY,
  proposal_id NVARCHAR(256),
  voter_id VARCHAR(100),
  vote VARCHAR(20),  -- accept, reject
  confidence FLOAT,
  voted_at DATETIME2,
  FOREIGN KEY (proposal_id) REFERENCES [FederatedProposals](proposal_id)
);
```

## Consensus Mechanism

### Voting Process

1. Node proposes pattern with confidence
2. Proposes to all peers with timeout
3. Peers vote accept/reject with confidence
4. Votes aggregated when quorum reached
5. Result determined by threshold

### Consensus Calculation

```
consensus_ratio = accept_votes / total_votes

has_consensus = (
  consensus_ratio >= threshold AND
  total_votes >= minPeersForConsensus
)
```

**Example with 3 peers (threshold 0.66, min 2):**
- 3 accepts: 3/3 = 1.0 ≥ 0.66 ✓ Consensus
- 2 accepts: 2/3 = 0.67 ≥ 0.66 ✓ Consensus
- 1 accept: 1/3 = 0.33 < 0.66 ✗ No consensus

### Network Partition Detection

```
is_partitioned = failed_syncs > successful_syncs
```

When partitioned:
- Local-first approach: patterns learned locally
- Eventual consistency: reconcile when reconnected
- No blocking on peer votes

## Use Cases

### Collaborative Learning

```javascript
// Node 1: Learn pattern locally
const pattern = { layer: 'conv', kernel: 5 };
await learner.addLocalPattern(pattern, 0.88);

// Propose to peers
const proposal = await learner.proposePeerUpdate(pattern);

// Peers vote
await otherLearner.voteOnProposal(proposal.proposalId, 'accept', 0.85);

// Check consensus
const consensus = await learner.getConsensus(proposal.proposalId);
if (consensus.hasConsensus) {
  // Adopt pattern across cluster
}
```

### Aggregation for Ensemble

```javascript
const patterns = await Promise.all([
  node1.getLocalPatterns(),
  node2.getLocalPatterns(),
  node3.getLocalPatterns()
]);

const flattened = patterns.flat();
const aggregated = await learner.aggregatePatterns(flattened);

// Use aggregated pattern in production
```

### Network Recovery

```javascript
// When network restores
const status = await learner.getPeerStatus();

if (status.onlinePeers === status.totalPeers) {
  // All nodes back online - reconcile
  await learner.syncWithPeers();
  
  // Re-propose any pending patterns
  for (const proposal of pendingProposals) {
    await learner.proposePeerUpdate(proposal);
  }
}
```

## Best Practices

1. **Set appropriate consensus threshold** (typically 0.66-0.8)
2. **Monitor peer status** regularly
3. **Aggregate patterns** before production use
4. **Handle network partitions gracefully**
5. **Verify pattern quality** before acceptance
6. **Log all proposals** for audit trail
7. **Sync periodically** for consistency
8. **Use confidence scores** to weight votes

## Error Handling

```javascript
try {
  const result = await learner.aggregatePatterns(patterns);
} catch (error) {
  if (error.message.includes('empty')) {
    console.error('No patterns to aggregate');
  } else {
    console.error('Aggregation failed:', error);
  }
}
```

## Performance Considerations

- Consensus mechanism: O(n) where n = peers
- Aggregation: O(n) where n = patterns
- Sync: Background operation, non-blocking
- Storage: Local patterns + proposals

## Integration Points

- **ModelVersioning**: Version agreed patterns
- **PatternDistribution**: Distribute consensus patterns
- **SystemAlerting**: Alert on consensus failures
- **SyncStrategy**: Sync consensus results

## Background Worker

Automatic synchronization if configured:

```javascript
const learner = new FederatedLearner(pool, cache, {
  syncIntervalMs: 30000  // Sync every 30 seconds
});

await learner.initialize(); // Starts worker

// Later: stop worker
learner.stopBackgroundSync();
```

## Limits and Defaults

| Parameter | Default | Max |
|-----------|---------|-----|
| Consensus threshold | 0.66 | 1.0 |
| Min peers | 2 | n_peers |
| Sync interval | 30s | N/A |
| Proposal TTL | 1h | N/A |
| Max active patterns | Unlimited | RAM |

## Eventual Consistency Model

The system guarantees:
1. **Local consistency**: Each node maintains valid patterns
2. **Eventual consistency**: Peers eventually agree
3. **Partition tolerance**: Works when split
4. **Automatic reconciliation**: Heals when reconnected

## Monitoring

Key metrics to monitor:
- `onlinePeers` / `totalPeers`: Network health
- `consensusesReached` / `proposalsMade`: Agreement rate
- `failedSyncs` / `syncAttempts`: Network reliability
- `averageConsensusTime`: Decision latency
