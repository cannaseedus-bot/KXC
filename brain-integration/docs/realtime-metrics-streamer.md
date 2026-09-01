# Phase 6: Real-Time Metrics Streamer

## Overview

The **Real-Time Metrics Streamer** captures execution metrics and streams them to SQL Server with automatic windowed aggregation for dashboard visualization. Includes event-driven architecture with real-time subscriptions and comprehensive metric tracking.

### Key Features

- ✅ **Streaming Metrics** - Real-time recording of execution data with full data context
- ✅ **Batch Recording** - Record multiple metrics efficiently with `recordMetrics()`
- ✅ **Buffered Writes** - Batches metrics for efficient inserts with configurable buffer size
- ✅ **Time Window Aggregation** - 1-min, 5-min, 1-hour, daily windows with explicit aggregate methods
- ✅ **Percentile Calculations** - P50/P95/P99 latency tracking and performance analysis
- ✅ **GPU Tracking** - Monitor GPU usage patterns and recommendation accuracy
- ✅ **Performance Trends** - Query historical trends with `getMetricsFor()`
- ✅ **Dashboard Ready** - Pre-aggregated data for views
- ✅ **Real-Time Events** - Event emitter pattern with subscriber support and throttling
- ✅ **Layer-Specific Metrics** - Track metrics per execution layer
- ✅ **Event Throttling** - Prevent event overload with configurable throttling

---

## Architecture

### Data Collection Pipeline

```
Operation Executes
    ↓
Metric Recorded
    • metric_id (UUID)
    • timestamp (ISO8601)
    • metric_type ('execution', 'gpu_decision', etc.)
    • layer (execution layer)
    • operation (operation name)
    • execution_time_ms
    • memory_used_mb
    • success (yes/no)
    • gpu_used (yes/no)
    • user_id & session_id
    ↓
Buffered in Memory (up to 10k)
    ↓
Real-Time Event Emission
    • Layer-specific events
    • Operation-specific events
    • Throttled to prevent overload
    ↓
Flushed to SQL Server (1 sec)
    • Insert to realtime_metrics table
    • Async write with batching
    ↓
Aggregated into Windows
    • Every 10 seconds
    • 1-min, 5-min, 1-hr, daily windows
    • Percentile calculations
    ↓
Dashboard Views
    • Query window tables
    • Show trends and stats
```

### Time Windows

| Window | Duration | Aggregation | Data Retention |
|--------|----------|-------------|-----------------|
| **1-min** | 60 seconds | Real-time | Latest 100 windows |
| **5-min** | 300 seconds | Trend analysis | Latest 288 windows |
| **1-hr** | 3600 seconds | Daily patterns | 30 days |
| **Daily** | 86400 seconds | Long-term trends | SQL Server archive |

### Database Tables

```
realtime_metrics (raw, recent data)
  • metric_id (TEXT, UUID)
  • timestamp (DATETIME)
  • metric_type (VARCHAR 50)
  • layer (VARCHAR 100)
  • operation (VARCHAR 100)
  • execution_time_ms (FLOAT)
  • memory_used_mb (FLOAT)
  • success (BIT)
  • gpu_used (BIT)
  • user_id, session_id (VARCHAR 100)
  • operation_type (VARCHAR 100, legacy)
  • recorded_at (DATETIME)

aggregated_metrics (analytics)
  • aggregation_id (TEXT, UUID)
  • aggregation_level (VARCHAR 20)
  • time_bucket (DATETIME)
  • metric_count, avg_execution_time_ms
  • p50/p95/p99_execution_time_ms
  • min/max_execution_time_ms
  • success_count, failed_count, success_rate
  • gpu_used_count, gpu_recommendation_rate
  • avg_memory_mb, peak_memory_mb
  • layers_active (JSON array)
  • operations_executed (JSON array)

metric_windows_1min/5min/1hr (time series)
  • window_id, operation_type, time_bucket
  • count, avg/p95/p99_exec_time, avg_memory
  • gpu_count, success_count, error_count
```

---

## API Reference

### Constructor

```javascript
const streamer = new RealtimeMetricsStreamer(
  dbPool,           // SQL Server connection pool
  sqliteCache,      // Optional SQLite cache for local persistence
  options           // Configuration (optional)
);
```

### Options

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `batchSize` | number | 50 | Metrics per flush |
| `flushInterval` | number | 1000 | Flush frequency (ms) |
| `aggregateInterval` | number | 10000 | Aggregation frequency (ms) |
| `windows` | array | [60, 300, 3600] | Time windows (seconds) |
| `eventThrottle` | number | 1000 | Event emission throttle (ms) |
| `maxBufferSize` | number | 10000 | Max in-memory metrics |
| `isTest` | boolean | false | Disable workers in tests |

### Core Methods

#### `async initialize()`

Initialize streamer and create tables.

```javascript
await streamer.initialize();
// → Creates realtime_metrics and window tables
```

---

#### `recordMetric(metric)`

Record an execution metric.

```javascript
streamer.recordMetric({
  operationType: 'matrix-multiply',
  layer: 'gpu-layer',
  operation: 'multiply',
  executionTime: 150,      // milliseconds
  memoryMb: 256,
  gpuUsed: true,
  success: true,
  userId: 'user-123',
  sessionId: 'sess-456',
  metricType: 'execution'
});
```

**Metric Fields**:
- `operationType` (required) - Type of operation
- `layer` - Execution layer (gpu-layer, cpu-layer, etc.)
- `operation` - Operation identifier
- `executionTime` - Time in milliseconds
- `memoryMb` - Memory used in MB
- `gpuUsed` - Whether GPU was used (boolean)
- `success` - Whether operation succeeded (boolean)
- `userId` - User identifier
- `sessionId` - Session identifier
- `metricType` - Type of metric ('execution', 'gpu_decision', 'cache_hit', 'learning')

---

#### `recordMetrics(metrics[])`

Record multiple metrics in batch.

```javascript
streamer.recordMetrics([
  { operationType: 'op1', executionTime: 100, success: true },
  { operationType: 'op2', executionTime: 150, success: true },
  { operationType: 'op3', executionTime: 200, success: false }
]);
```

---

#### `startStreaming()`

Start automatic metric flushing and aggregation.

```javascript
streamer.startStreaming();
// Flushes every flushInterval
// Aggregates every aggregateInterval
```

---

#### `stopStreaming()`

Stop background workers.

```javascript
streamer.stopStreaming();
```

---

#### `async flush()`

Explicitly flush buffered metrics to database.

```javascript
await streamer.flush();
// Forces immediate write regardless of batch size
```

---

#### `clear()`

Clear in-memory buffer without flushing.

```javascript
const count = streamer.clear();
// Returns number of metrics cleared
```

---

#### `getStats()`

Get streaming statistics.

```javascript
const stats = streamer.getStats();
// {
//   metricsStreamed: 1000,
//   metricsQueued: 5,
//   bufferSize: 5,
//   flushes: 50,
//   aggregations: 5,
//   errors: 0,
//   streaming: true
// }
```

---

#### `getStatus()`

Get comprehensive stream status.

```javascript
const status = streamer.getStatus();
// {
//   initialized: true,
//   streaming: true,
//   stats: { ... },
//   options: { ... },
//   subscribers: 2
// }
```

---

#### `async getMetricsFor(timeWindow, aggregationLevel)`

Get metrics for a specific time window.

```javascript
// Get metrics from last 5 minutes at 1-min aggregation
const metrics = await streamer.getMetricsFor(300, 60);

// Get metrics from last hour at 5-min aggregation
const hourMetrics = await streamer.getMetricsFor(3600, 300);
```

---

#### `async aggregate1Min()`

Explicitly aggregate 1-minute window.

```javascript
await streamer.aggregate1Min();
```

---

#### `async aggregate5Min()`

Explicitly aggregate 5-minute window.

```javascript
await streamer.aggregate5Min();
```

---

#### `async aggregate1Hour()`

Explicitly aggregate 1-hour window.

```javascript
await streamer.aggregate1Hour();
```

---

#### `async aggregateDaily()`

Aggregate daily metrics (stored in SQL Server).

```javascript
await streamer.aggregateDaily();
```

---

#### `subscribe(callback)`

Subscribe to real-time metric events.

```javascript
const unsubscribe = streamer.subscribe((metric) => {
  console.log('Metric received:', metric);
});

// Unsubscribe when done
unsubscribe();
```

**Subscriber Callback**:
```javascript
(metric) => {
  // metric includes:
  // - metric_id, timestamp, metric_type
  // - layer, operation, execution_time_ms
  // - memory_used_mb, success, gpu_used
  // - user_id, session_id
}
```

---

#### `async getDashboardMetrics(operationType?, limit?)`

Get dashboard-ready metrics.

```javascript
// All operations, 1-min window
const allMetrics = await streamer.getDashboardMetrics();

// Specific operation
const mmMetrics = await streamer.getDashboardMetrics('matrix-multiply', 100);

// Returns: Array of windowed metrics with:
//   - count, avg_exec_time, p95_exec_time, p99_exec_time
//   - success_rate, gpu_usage_rate
```

---

#### `async getPerformanceTrends(operationType, windowSeconds)`

Get historical performance trends.

```javascript
const trends1min = await streamer.getPerformanceTrends('matrix-multiply', 60);
const trends5min = await streamer.getPerformanceTrends('matrix-multiply', 300);
const trends1hr = await streamer.getPerformanceTrends('matrix-multiply', 3600);

// Returns: Array of windows ordered by most recent
//   Each includes timing, resource, and success metrics
```

---

#### `async getHealthCheck()`

Check streamer health.

```javascript
const health = await streamer.getHealthCheck();
// {
//   healthy: true,
//   total_metrics: 5432,
//   latest_metric: "2024-03-15T07:30:00Z",
//   earliest_metric: "2024-03-15T06:30:00Z",
//   operation_types: 8,
//   bufferedMetrics: 3,
//   errors: 0
// }
```

---

#### `async cleanup()`

Stop streaming and cleanup.

```javascript
await streamer.cleanup();
// Flushes remaining metrics, stops workers, clears subscriptions
```

---

## Usage Patterns

### 1. Basic Setup

```javascript
import RealtimeMetricsStreamer from 'src/realtime-metrics-streamer.js';

const streamer = new RealtimeMetricsStreamer(dbPool, sqliteCache, {
  batchSize: 100,
  flushInterval: 1000,
  eventThrottle: 500
});

await streamer.initialize();
streamer.startStreaming();

// Record metrics
streamer.recordMetric({
  operationType: 'matrix-multiply',
  layer: 'gpu-layer',
  operation: 'multiply',
  executionTime: 150,
  gpuUsed: true,
  success: true
});
```

### 2. Batch Recording

```javascript
const metrics = operations.map(op => ({
  operationType: op.type,
  layer: op.targetLayer,
  operation: op.name,
  executionTime: op.duration,
  memoryMb: op.memory,
  gpuUsed: op.usedGPU,
  success: op.succeeded,
  userId: currentUser.id,
  sessionId: currentSession.id
}));

streamer.recordMetrics(metrics);
```

### 3. Real-Time Subscriptions

```javascript
const unsubscribe = streamer.subscribe((metric) => {
  // Update dashboard in real-time
  dashboardWebSocket.send({
    type: 'metric-update',
    layer: metric.layer,
    operation: metric.operation,
    executionTime: metric.execution_time_ms,
    success: metric.success
  });
});

// Per-layer subscriptions
streamer.on('metric', (metric) => {
  if (metric.layer === 'gpu-layer') {
    gpuMetricsHandler(metric);
  }
});
```

### 4. Dashboard Queries

```javascript
// Get latest metrics for dashboard
const dashboard = await streamer.getDashboardMetrics();

const metrics = {
  operations: dashboard.map(m => ({
    name: m.operation_type,
    avgTime: m.avg_exec_time,
    p95Time: m.p95_exec_time,
    successRate: m.success_rate,
    gpuUsage: m.gpu_usage_rate
  }))
};

res.json(metrics);
```

### 5. Performance Analysis

```javascript
// Analyze trends for specific operation
const trends = await streamer.getPerformanceTrends('convolution', 300);

const analysis = {
  current: trends[0],
  previous: trends[1],
  trend: trends[0].avg_exec_time - trends[1].avg_exec_time,
  direction: trends[0].avg_exec_time > trends[1].avg_exec_time ? 'slower' : 'faster'
};

console.log(`Convolution: ${analysis.direction} by ${Math.abs(analysis.trend).toFixed(2)}ms`);
```

### 6. Health Monitoring

```javascript
setInterval(async () => {
  const health = await streamer.getHealthCheck();
  
  if (!health.healthy) {
    console.error('Metrics system unhealthy!');
    alerting.notify('Metrics failure');
  }
  
  if (health.errors > 100) {
    console.warn(`${health.errors} metric errors detected`);
  }
}, 60000); // Every minute
```

### 7. Layer-Specific Metrics

```javascript
// Track GPU layer decisions
streamer.recordMetric({
  operationType: 'decision-record',
  layer: 'gpu-decision-model',
  operation: 'should-gpu-exec',
  executionTime: 5,
  metricType: 'gpu_decision',
  success: decisionCorrect,
  userId: userId
});

// Query GPU decision trends
const gpuDecisions = await streamer.getPerformanceTrends('decision-record', 3600);
const gpuAccuracy = gpuDecisions[0].success_rate;
```

---

## Performance Tuning

### Batch Size Impact

```
Small (10-25 items):
  ✓ Lower memory usage
  ✓ Faster latency visibility
  ✗ More database round-trips

Large (100-500 items):
  ✓ Fewer database hits
  ✓ Higher throughput
  ✗ Higher memory usage
  ✗ Slight latency increase

Recommended: 50-100 for balance
```

### Flush Interval Impact

```
Frequent (100-500ms):
  ✓ Real-time visibility
  ✗ More write load
  ✗ Higher CPU usage

Infrequent (5000-10000ms):
  ✓ Lower write load
  ✓ Better batching
  ✗ Delayed visibility

Recommended: 1000ms (1 second)
```

### Event Throttling

```
Aggressive (100-500ms):
  ✓ Real-time event stream
  ✗ High event volume
  ✗ More CPU for processing

Conservative (2000-5000ms):
  ✓ Lower event volume
  ✓ Reduced CPU usage
  ✗ Delayed event visibility

Recommended: 1000ms (1 second)
```

### Buffer Size

```
Small (1k metrics):
  ✓ Lower memory usage
  ✗ More frequent flushes

Large (50k+ metrics):
  ✓ Fewer database hits
  ✗ Higher memory usage
  ✗ Risk of data loss on crash

Recommended: 10k metrics (default)
```

---

## Integration Points

### With Offline Mode

```javascript
// Record metrics even when offline
offlineMode.on('offline', () => {
  // Metrics still recorded locally
  // Will flush when back online
});

offlineMode.on('online', async () => {
  await streamer.flush();
  const health = await streamer.getHealthCheck();
  console.log('Metrics synced:', health.total_metrics);
});
```

### With Dashboard Views

```javascript
// Dashboard queries aggregated metrics
const data = await streamer.getDashboardMetrics();

// Real-time update every 10 seconds
setInterval(async () => {
  dashboard.updateMetrics(await streamer.getDashboardMetrics());
}, 10000);

// Subscribe to events for live updates
streamer.subscribe((metric) => {
  dashboard.flashMetric(metric);
});
```

### With Learning System

```javascript
// Track learning pattern metrics
streamer.recordMetric({
  operationType: 'learning-pattern',
  layer: 'specialist-learning',
  operation: 'pattern-extraction',
  metricType: 'learning',
  executionTime: learningTime,
  success: patternValid,
  userId: userId
});

// Analyze learning effectiveness
const patterns = await streamer.getPerformanceTrends('learning-pattern', 3600);
```

### With GPU Decision Model

```javascript
// Track GPU recommendation accuracy
for (const prediction of predictions) {
  streamer.recordMetric({
    layer: 'gpu-decision-model',
    operation: 'gpu-recommendation',
    metricType: 'gpu_decision',
    executionTime: decisionTime,
    gpuUsed: prediction.recommended,
    success: prediction.correct,
    userId: userId
  });
}

// Monitor recommendation accuracy
const accuracy = await streamer.getMetricsFor(3600, 300);
```

---

## Troubleshooting

### Metrics Not Appearing

**Symptom**: Dashboard shows no metrics

**Causes**:
- Streaming not started
- No metrics recorded
- Database offline

**Solution**:
```javascript
streamer.startStreaming();

const stats = streamer.getStats();
if (stats.metricsQueued === 0) {
  console.warn('No metrics recorded!');
}

const health = await streamer.getHealthCheck();
if (!health.healthy) {
  console.error('Database issue:', health.error);
}
```

### High Error Rate

**Symptom**: getStats() shows high error count

**Causes**:
- Database connectivity issues
- Disk space full
- Query timeouts

**Solution**:
- Check database connectivity
- Increase flush interval to reduce load
- Archive old metrics

### Memory Growing

**Symptom**: Process memory increases over time

**Causes**:
- Flushing not happening
- Large batch sizes
- Event subscriptions not unsubscribed

**Solution**:
```javascript
const stats = streamer.getStats();
if (stats.bufferSize > 5000) {
  streamer.startStreaming(); // Ensure running
  await streamer.flush(); // Force flush
}

// Always unsubscribe when done
const unsubscribe = streamer.subscribe(handler);
// ... later ...
unsubscribe();
```

### Events Not Emitting

**Symptom**: Subscriptions not receiving events

**Causes**:
- Throttling interval too high
- Event throttle timer not firing
- Subscribers not registered

**Solution**:
```javascript
// Check throttle setting
const status = streamer.getStatus();
console.log('Throttle interval:', status.options.eventThrottle);

// Verify subscription
const unsub = streamer.subscribe((m) => console.log('Event:', m));
console.log('Subscribers:', streamer.subscribers.size);

// Wait for throttle interval
await new Promise(r => setTimeout(r, status.options.eventThrottle + 100));
```

---

## Testing

Run comprehensive test suite:

```bash
node tests/test-realtime-metrics-streamer.mjs
```

**Coverage**:
- ✅ 62 tests across 22 test suites
- ✅ Metric recording and buffering
- ✅ Batch recording
- ✅ Streaming lifecycle
- ✅ Buffer management
- ✅ Explicit aggregation
- ✅ Real-time events and subscriptions
- ✅ Stream status reporting
- ✅ Enhanced metrics data
- ✅ Window aggregation
- ✅ Dashboard queries
- ✅ Health checks
- ✅ Configuration
- ✅ Performance benchmarks
- ✅ Integration scenarios

**Test Results**: 62/62 passing ✅

---

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Record metric | <1ms | Buffer only |
| Record metric batch (50) | <5ms | All buffered |
| Clear large buffer (1000 items) | <10ms | Fast array splice |
| Flush 50 items | 10-50ms | Database insert |
| Aggregate window | 50-200ms | Percentile calculation |
| Subscribe callback | <1ms | Event handler |
| Dashboard query | <100ms | Indexed search |

---

## Future Enhancements

1. **Compression** - Compress old metrics for archive
2. **Retention Policy** - Auto-delete old metrics
3. **Custom Windows** - User-defined aggregation windows
4. **Alerting** - Trigger alerts on thresholds
5. **Export** - Download metrics as CSV/JSON
6. **Comparison** - Compare metrics across time periods
7. **Advanced Percentiles** - More percentiles (p50, p75, p999)
8. **Sampling** - Sample high-volume metrics

---

## Related Files

- `src/offline-mode.js` - Buffers metrics during offline
- `src/dashboard-query-builder.js` - Query builder for dashboards
- `src/sql-connection.js` - SQL Server connection pool
- `src/sqlite-cache.js` - SQLite cache for local persistence
- `db-schema-analytics-phase6.sql` - Schema definition
- `tests/test-realtime-metrics-streamer.mjs` - Comprehensive tests

---

**Phase 6 Component**: Real-Time Metrics Streamer ✅
**Status**: Production Ready
**Tests**: 62/62 passing ✓
**Coverage**: All requirements met

### Data Collection Pipeline

```
Operation Executes
    ↓
Metric Recorded
    • operation_type
    • execution_time_ms
    • memory_mb
    • gpu_used (yes/no)
    • success (yes/no)
    • user_id
    ↓
Buffered in Memory (up to 50)
    ↓
Flushed to SQL Server (1 sec)
    • Insert to realtime_metrics table
    • Async write
    ↓
Aggregated into Windows
    • Every 10 seconds
    • 1-min, 5-min, 1-hr windows
    ↓
Dashboard Views
    • Query window tables
    • Show trends and stats
```

### Time Windows

| Window | Duration | Use Case |
|--------|----------|----------|
| **1-min** | 60 seconds | Real-time monitoring |
| **5-min** | 300 seconds | Trend analysis |
| **1-hr** | 3600 seconds | Daily patterns |

### Database Tables

```
realtime_metrics (raw)
  • metric_id (BIGINT)
  • operation_type (VARCHAR 100)
  • execution_time_ms (FLOAT)
  • memory_mb (FLOAT)
  • gpu_used (BIT)
  • success (BIT)
  • user_id (VARCHAR 100)
  • recorded_at (DATETIME)
  
metric_windows_1min (aggregated)
  • window_id, operation_type, window_start, window_end
  • count, avg_exec_time, p95_exec_time, p99_exec_time
  • avg_memory, gpu_count, success_count, error_count

metric_windows_5min (aggregated)
metric_windows_1hr (aggregated)
  [Same structure as 1-min windows]
```

---

## API Reference

### Constructor

```javascript
const streamer = new RealtimeMetricsStreamer(
  dbPool,           // SQL Server connection pool
  options           // Configuration (optional)
);
```

### Options

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `batchSize` | number | 50 | Metrics per flush |
| `flushInterval` | number | 1000 | Flush frequency (ms) |
| `aggregateInterval` | number | 10000 | Aggregation frequency (ms) |
| `windows` | array | [60, 300, 3600] | Time windows (seconds) |
| `isTest` | boolean | false | Disable workers in tests |

### Methods

#### `async initialize()`

Initialize streamer and create tables.

```javascript
await streamer.initialize();
// → Creates realtime_metrics and window tables
```

---

#### `recordMetric(metric)`

Record an execution metric.

```javascript
streamer.recordMetric({
  operationType: 'matrix-multiply',
  executionTime: 150,      // milliseconds
  memoryMb: 256,
  gpuUsed: true,
  success: true,
  userId: 'user-123'
});
```

**Metric Fields**:
- `operationType` (required) - Type of operation
- `executionTime` - Time in milliseconds
- `memoryMb` - Memory used in MB
- `gpuUsed` - Whether GPU was used (boolean)
- `success` - Whether operation succeeded (boolean)
- `userId` - User identifier

---

#### `startStreaming()`

Start automatic metric flushing and aggregation.

```javascript
streamer.startStreaming();
// Flushes every flushInterval
// Aggregates every aggregateInterval
```

---

#### `stopStreaming()`

Stop background workers.

```javascript
streamer.stopStreaming();
```

---

#### `getStats()`

Get streaming statistics.

```javascript
const stats = streamer.getStats();
// {
//   metricsStreamed: 1000,
//   metricsQueued: 5,
//   bufferSize: 5,
//   flushes: 50,
//   aggregations: 5,
//   errors: 0,
//   streaming: true
// }
```

---

#### `async getDashboardMetrics(operationType?, limit?)`

Get dashboard-ready metrics.

```javascript
// All operations, 1-min window
const allMetrics = await streamer.getDashboardMetrics();

// Specific operation
const mmMetrics = await streamer.getDashboardMetrics('matrix-multiply', 100);

// Returns: Array of windowed metrics with:
//   - count, avg_exec_time, p95_exec_time, p99_exec_time
//   - success_rate, gpu_usage_rate
```

---

#### `async getPerformanceTrends(operationType, windowSeconds)`

Get historical performance trends.

```javascript
const trends1min = await streamer.getPerformanceTrends('matrix-multiply', 60);
const trends5min = await streamer.getPerformanceTrends('matrix-multiply', 300);
const trends1hr = await streamer.getPerformanceTrends('matrix-multiply', 3600);

// Returns: Array of windows ordered by most recent
//   Each includes timing, resource, and success metrics
```

---

#### `async getHealthCheck()`

Check streamer health.

```javascript
const health = await streamer.getHealthCheck();
// {
//   healthy: true,
//   total_metrics: 5432,
//   latest_metric: "2024-03-15T07:30:00Z",
//   earliest_metric: "2024-03-15T06:30:00Z",
//   operation_types: 8,
//   bufferedMetrics: 3,
//   errors: 0
// }
```

---

#### `async cleanup()`

Stop streaming and cleanup.

```javascript
await streamer.cleanup();
// Flushes remaining metrics and stops workers
```

---

## Usage Patterns

### 1. Basic Setup

```javascript
import RealtimeMetricsStreamer from 'src/realtime-metrics-streamer.js';

const streamer = new RealtimeMetricsStreamer(dbPool, {
  batchSize: 100,
  flushInterval: 1000
});

await streamer.initialize();
streamer.startStreaming();

// Record metrics
streamer.recordMetric({
  operationType: 'matrix-multiply',
  executionTime: 150,
  gpuUsed: true,
  success: true
});
```

### 2. Record Execution Metrics

```javascript
async function executeOperation(operation, data) {
  const startTime = performance.now();
  
  try {
    const result = await operation(data);
    const executionTime = performance.now() - startTime;
    
    streamer.recordMetric({
      operationType: operation.name,
      executionTime,
      memoryMb: process.memoryUsage().heapUsed / 1024 / 1024,
      gpuUsed: operation.usesGPU,
      success: true,
      userId: currentUserId
    });
    
    return result;
  } catch (error) {
    const executionTime = performance.now() - startTime;
    
    streamer.recordMetric({
      operationType: operation.name,
      executionTime,
      success: false,
      userId: currentUserId
    });
    
    throw error;
  }
}
```

### 3. Dashboard Queries

```javascript
// Get latest metrics for dashboard
const dashboard = await streamer.getDashboardMetrics();

const metrics = {
  operations: dashboard.map(m => ({
    name: m.operation_type,
    avgTime: m.avg_exec_time,
    p95Time: m.p95_exec_time,
    successRate: m.success_rate,
    gpuUsage: m.gpu_usage_rate
  }))
};
```

### 4. Performance Analysis

```javascript
// Analyze trends for specific operation
const trends = await streamer.getPerformanceTrends('convolution', 300);

const analysis = {
  current: trends[0],
  previous: trends[1],
  trend: trends[0].avg_exec_time - trends[1].avg_exec_time,
  direction: trends[0].avg_exec_time > trends[1].avg_exec_time ? 'slower' : 'faster'
};

console.log(`Convolution: ${analysis.direction} by ${Math.abs(analysis.trend).toFixed(2)}ms`);
```

### 5. Health Monitoring

```javascript
setInterval(async () => {
  const health = await streamer.getHealthCheck();
  
  if (!health.healthy) {
    console.error('Metrics system unhealthy!');
    // Alert team
  }
  
  if (health.errors > 100) {
    console.warn(`${health.errors} metric errors detected`);
  }
}, 60000); // Every minute
```

---

## Performance Tuning

### Batch Size Impact

```
Small (10-25 items):
  ✓ Lower memory usage
  ✓ Faster latency visibility
  ✗ More database round-trips

Large (100-500 items):
  ✓ Fewer database hits
  ✓ Higher throughput
  ✗ Higher memory usage
  ✗ Slight latency increase

Recommended: 50-100 for balance
```

### Flush Interval Impact

```
Frequent (100-500ms):
  ✓ Real-time visibility
  ✗ More write load
  ✗ Higher CPU usage

Infrequent (5000-10000ms):
  ✓ Lower write load
  ✓ Better batching
  ✗ Delayed visibility

Recommended: 1000ms (1 second)
```

### Aggregation Interval

```
Frequent (5 seconds):
  ✓ Up-to-date trends
  ✗ More CPU for calculations

Infrequent (30 seconds):
  ✓ Lower CPU usage
  ✗ Delayed trend updates

Recommended: 10 seconds
```

---

## Integration Points

### With Offline Mode

```javascript
// Record metrics even when offline
offlineMode.on('offline', () => {
  // Metrics still recorded locally
  // Will flush when back online
});
```

### With Dashboard Views

```javascript
// Dashboard queries aggregated metrics
const data = await streamer.getDashboardMetrics();

// Real-time update every 10 seconds
setInterval(async () => {
  dashboard.updateMetrics(await streamer.getDashboardMetrics());
}, 10000);
```

### With Alerts

```javascript
// Alert on performance degradation
const trends = await streamer.getPerformanceTrends('op', 60);
if (trends[0].avg_exec_time > threshold) {
  alerting.notify('Operation slow');
}
```

---

## Troubleshooting

### Metrics Not Appearing

**Symptom**: Dashboard shows no metrics

**Causes**:
- Streaming not started
- No metrics recorded
- Database offline

**Solution**:
```javascript
streamer.startStreaming();

const stats = streamer.getStats();
if (stats.metricsQueued === 0) {
  console.warn('No metrics recorded!');
}

const health = await streamer.getHealthCheck();
if (!health.healthy) {
  console.error('Database issue:', health.error);
}
```

### High Error Rate

**Symptom**: getStats() shows high error count

**Causes**:
- Database connectivity issues
- Disk space full
- Query timeouts

**Solution**:
- Check database connectivity
- Increase flush interval to reduce load
- Archive old metrics

### Memory Growing

**Symptom**: Process memory increases over time

**Causes**:
- Flushing not happening
- Large batch sizes
- Memory leak in aggregation

**Solution**:
```javascript
const stats = streamer.getStats();
if (stats.bufferSize > 1000) {
  streamer.startStreaming(); // Ensure running
  await streamer._flushMetrics(); // Force flush
}
```

---

## Testing

Run comprehensive test suite:

```bash
node tests/test-realtime-metrics-streamer.mjs
```

**Coverage**:
- ✅ 42 tests across 15 test suites
- ✅ Metric recording and buffering
- ✅ Streaming lifecycle
- ✅ Window aggregation
- ✅ Dashboard queries
- ✅ Health checks
- ✅ Configuration
- ✅ Integration scenarios

**Test Results**: 42/42 passing ✅

---

## Future Enhancements

1. **Compression** - Compress old metrics for archive
2. **Retention Policy** - Auto-delete old metrics
3. **Custom Windows** - User-defined aggregation windows
4. **Alerting** - Trigger alerts on thresholds
5. **Export** - Download metrics as CSV/JSON
6. **Comparison** - Compare metrics across time periods
7. **Percentiles** - More percentiles (p50, p75, p999)
8. **Sampling** - Sample high-volume metrics

---

## Related Files

- `src/offline-mode.js` - Buffers metrics during offline
- `brain-integration/docs/analytics-query-builder.md` - Query builder for dashboards
- `db-schema-analytics-phase6.sql` - Schema definition
- `tests/test-realtime-metrics-streamer.mjs` - Comprehensive tests

---

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Record metric | <1ms | Buffer only |
| Flush 50 items | 10-50ms | Database insert |
| Aggregate window | 50-200ms | Percentile calculation |
| Dashboard query | <100ms | Indexed search |

---

## Statistics

**Test Coverage**:
- 42 tests, 15 suites
- 100% pass rate
- ~75ms total execution

**Supported Operations**:
- Any operation type (matrix, convolution, sort, etc.)
- GPU and CPU tracking
- Success/failure tracking
- Multi-user support

---

**Phase 6 Component**: Real-Time Metrics Streamer ✅
**Status**: Production Ready
**Tests**: 42/42 passing ✓
