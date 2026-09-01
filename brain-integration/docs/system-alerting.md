# SystemAlerting API Documentation

## Overview

The SystemAlerting module provides real-time system health monitoring with configurable alerts, multiple severity levels, acknowledgment tracking, and temporary silencing. It integrates with pattern learning and federated systems to detect anomalies.

## Installation

```javascript
import SystemAlerting from '../src/system-alerting.js';

const alerting = new SystemAlerting(dbPool, sqliteCache, {
  checkIntervalMs: 10000,           // Check every 10 seconds
  hysteresisWindow: 3,              // Require 3 consecutive triggers
  maxAlertHistory: 1000,
  notificationChannels: ['log', 'event', 'webhook'],
  webhookUrl: 'https://alerts.example.com/webhook'
});

await alerting.initialize();
```

## Class Constructor

```javascript
SystemAlerting(dbPool, sqliteCache, options = {})
```

### Parameters

- **dbPool** (DBPool): SQL Server connection pool
- **sqliteCache** (SQLiteCache): SQLite cache
- **options** (Object):
  - `checkIntervalMs` (number): Alert check frequency. Default: `10000`
  - `hysteresisWindow` (number): Consecutive triggers to activate. Default: `3`
  - `maxAlertHistory` (number): Max history entries. Default: `1000`
  - `notificationChannels` (Array): Notification methods. Default: `['log']`
  - `webhookUrl` (string): Webhook endpoint for alerts

## Methods

### initialize()

Initializes the alerting system.

```javascript
const result = await alerting.initialize();
// Returns: { initialized: true }
```

### registerAlert(name, condition, threshold, action)

Registers a new alert condition.

```javascript
const result = await alerting.registerAlert(
  'High GPU Memory Usage',
  (metrics) => {
    const gpuMem = metrics.get('gpu_memory_percent');
    return gpuMem && gpuMem[gpuMem.length - 1].value > 85;
  },
  85,
  {
    severity: 'critical',
    action: 'log'
  }
);

// Returns: {
//   alertId: 'alert-1699123456789-xyz789',
//   registered: true
// }
```

**Parameters:**
- `name` (string): Alert display name
- `condition` (Function): Returns boolean indicating alert state
- `threshold` (number): Threshold value
- `action` (Object):
  - `severity` (string): `'info'`, `'warning'`, or `'critical'`. Default: `'warning'`
  - `action` (string): Action to take

**Returns:** Alert registration with unique `alertId`

**Throws:**
- `Error` if invalid severity level

### recordMetric(metricName, value, timestamp)

Records a metric value for alert evaluation.

```javascript
await alerting.recordMetric('cpu_usage', 65.5);
await alerting.recordMetric('memory_percent', 42.1);
await alerting.recordMetric('gpu_decision_accuracy', 0.94);

// Custom timestamp (ms since epoch)
await alerting.recordMetric('latency_ms', 150, Date.now() - 5000);
```

**Parameters:**
- `metricName` (string): Metric identifier
- `value` (number): Metric value
- `timestamp` (number): Optional Unix timestamp. Default: `now`

**Returns:** Result object with value and timestamp

**Throws:**
- `Error` if invalid metric name
- `Error` if value not numeric

### checkAlerts()

Evaluates all registered alert conditions.

```javascript
const result = await alerting.checkAlerts();

// Returns: {
//   triggered: 2,    // Alerts activated
//   resolved: 1,     // Alerts deactivated
//   duration: 45     // Check time in ms
// }
```

**Automatically Called By:**
- Background worker (if configured)
- Can be called manually

**Returns:** Result with triggered/resolved counts

### getAlertStatus()

Gets current alert status across all alerts.

```javascript
const status = await alerting.getAlertStatus();

// Returns: {
//   totalAlerts: 15,
//   activeAlerts: 2,
//   silencedAlerts: 1,
//   acknowledgedAlerts: 1,
//   byLevel: {
//     critical: 1,
//     warning: 1,
//     info: 0
//   },
//   details: [
//     {
//       alert_id: 'alert-1699123456789-xyz789',
//       alert_name: 'High GPU Memory',
//       severity: 'critical',
//       triggered_at: 1699123456000,
//       current_value: 92,
//       threshold: 85,
//       acknowledged: false
//     },
//     ...
//   ]
// }
```

**Returns:** Comprehensive alert status object

### acknowledgeAlert(alertId)

Manually acknowledge an active alert.

```javascript
const result = await alerting.acknowledgeAlert('alert-1699123456789-xyz789');

// Returns: {
//   success: true,
//   alertId: 'alert-1699123456789-xyz789',
//   acknowledgedAt: 1699123456890
// }
```

**Purpose:**
- Record that alert was seen
- Mark for human follow-up
- Separate from resolution

**Returns:** Acknowledgment confirmation

**Throws:**
- `Error` if alert not found

### silenceAlert(alertId, durationMs)

Temporarily silence an alert.

```javascript
// Silence for 1 hour
const result = await alerting.silenceAlert(
  'alert-1699123456789-xyz789',
  3600000
);

// Returns: {
//   success: true,
//   alertId: 'alert-1699123456789-xyz789',
//   silencedUntil: 1699127056000
// }
```

**Use Cases:**
- Maintenance windows
- Known issues being worked on
- Test/staging false positives

**Parameters:**
- `alertId` (string): Alert to silence
- `durationMs` (number): Silence duration in milliseconds

**Returns:** Silence confirmation with expiration time

**Throws:**
- `Error` if invalid duration

### getAlertHistory(limit)

Gets recent alert events.

```javascript
const history = await alerting.getAlertHistory(100);

// Returns: [
//   {
//     history_id: 'hist-...',
//     alert_id: 'alert-...',
//     alert_name: 'High GPU Memory',
//     event_type: 'triggered',  // or 'resolved'
//     occurred_at: 2024-01-01T12:00:00Z,
//     value: 92,
//     threshold: 85
//   },
//   ...
// ]
```

**Parameters:**
- `limit` (number): Max entries. Default: `50`

**Returns:** Array of history entries (newest first)

### getMetrics(metricName, limit)

Gets recorded metric values.

```javascript
const cpuMetrics = await alerting.getMetrics('cpu_usage', 100);

// Returns: [
//   { value: 45.5, timestamp: 1699123456000 },
//   { value: 48.2, timestamp: 1699123457000 },
//   { value: 52.1, timestamp: 1699123458000 },
//   ...
// ]
```

**Parameters:**
- `metricName` (string): Metric to retrieve
- `limit` (number): Max data points. Default: `100`

**Returns:** Array of metric data points

### getStats()

Gets alerting system statistics.

```javascript
const stats = alerting.getStats();

// Returns: {
//   alertsTriggered: 12,
//   alertsResolved: 10,
//   alertsAcknowledged: 8,
//   metricsRecorded: 5240,
//   checkCycles: 524,
//   avgResponseTime: 42,        // ms
//   totalAlerts: 15,
//   activeAlerts: 2,
//   silencedAlerts: 1,
//   historySize: 450
// }
```

## Alert Types

### Performance Degradation

Detects performance issues:

```javascript
// Latency alert
await alerting.registerAlert(
  'High Latency',
  (metrics) => {
    const latency = metrics.get('latency_ms');
    return latency && latency[latency.length - 1].value > 500;
  },
  500,
  { severity: 'warning' }
);

// Success rate alert
await alerting.registerAlert(
  'Low Success Rate',
  (metrics) => {
    const rate = metrics.get('success_rate');
    return rate && rate[rate.length - 1].value < 0.95;
  },
  0.95,
  { severity: 'warning' }
);

// GPU accuracy alert
await alerting.registerAlert(
  'GPU Accuracy Drop',
  (metrics) => {
    const accuracy = metrics.get('gpu_decision_accuracy');
    return accuracy && accuracy[accuracy.length - 1].value < 0.85;
  },
  0.85,
  { severity: 'critical' }
);
```

### System Health

Monitors infrastructure health:

```javascript
// Offline duration
await alerting.registerAlert(
  'Extended Offline Period',
  (metrics) => {
    const offline = metrics.get('offline_duration_ms');
    return offline && offline[offline.length - 1].value > 300000; // 5 min
  },
  300000,
  { severity: 'critical' }
);

// Cache hit rate
await alerting.registerAlert(
  'Low Cache Hit Rate',
  (metrics) => {
    const rate = metrics.get('cache_hit_rate');
    return rate && rate[rate.length - 1].value < 0.70;
  },
  0.70,
  { severity: 'warning' }
);
```

### Resource Constraints

Detects resource issues:

```javascript
// Memory usage
await alerting.registerAlert(
  'High Memory Usage',
  (metrics) => {
    const mem = metrics.get('memory_percent');
    return mem && mem[mem.length - 1].value > 80;
  },
  80,
  { severity: 'warning' }
);

// GPU queue
await alerting.registerAlert(
  'GPU Queue Backlog',
  (metrics) => {
    const queue = metrics.get('gpu_queue_length');
    return queue && queue[queue.length - 1].value > 1000;
  },
  1000,
  { severity: 'critical' }
);
```

### Error Conditions

Detects exceptions:

```javascript
// Exception count
await alerting.registerAlert(
  'High Exception Rate',
  (metrics) => {
    const errors = metrics.get('exception_count');
    return errors && errors[errors.length - 1].value > 10;
  },
  10,
  { severity: 'critical' }
);

// Model rollback
await alerting.registerAlert(
  'Model Rollback Detected',
  (metrics) => {
    const rollbacks = metrics.get('model_rollback_count');
    return rollbacks && rollbacks[rollbacks.length - 1].value > 0;
  },
  0,
  { severity: 'critical' }
);
```

## Database Schema

### SystemAlerts table

```sql
CREATE TABLE [SystemAlerts] (
  alert_id NVARCHAR(128) PRIMARY KEY,
  alert_name VARCHAR(100),
  severity VARCHAR(20),          -- info, warning, critical
  condition_json NVARCHAR(MAX),
  threshold_value FLOAT,
  current_value FLOAT,
  triggered BIT,
  triggered_at DATETIME2,
  acknowledged_at DATETIME2,
  silenced_until DATETIME2,
  action_taken VARCHAR(500),
  resolved_at DATETIME2,
  trigger_count INT
);
```

### AlertHistory table

```sql
CREATE TABLE [AlertHistory] (
  history_id NVARCHAR(256) PRIMARY KEY,
  alert_id NVARCHAR(128),
  alert_name VARCHAR(100),
  event_type VARCHAR(50),        -- triggered, resolved
  occurred_at DATETIME2,
  value FLOAT,
  threshold FLOAT
);
```

## Hysteresis

Prevents flapping by requiring consecutive triggers:

```javascript
hysteresisWindow: 3  // Require 3 consecutive high values
```

**Example:**
- Alert threshold: CPU > 80%
- Metric values: 78, 82, 85, 86, 79, 78, 72
- Hysteresis counter: 0, 1, 2, 3 (triggered), 2, 1, 0 (resolved)

## Use Cases

### Production Monitoring

```javascript
// Create comprehensive monitoring
await alerting.registerAlert('High Latency', latencyCondition, 500, {
  severity: 'warning'
});

await alerting.registerAlert('GPU Memory Critical', gpuMemCondition, 90, {
  severity: 'critical'
});

await alerting.registerAlert('Model Accuracy Drop', accuracyCondition, 0.85, {
  severity: 'critical'
});

// Start background monitoring
// (automatic via checkWorker)
```

### Maintenance Window

```javascript
// Silence predictable alerts during deployment
const alerts = await alerting.getAlertStatus();

for (const alert of alerts.details) {
  if (alert.severity === 'warning') {
    await alerting.silenceAlert(alert.alert_id, 300000); // 5 min
  }
}

// Deploy changes
await deployNewVersion();

// Silence expires automatically
```

### Incident Response

```javascript
// Get full context during incident
const status = await alerting.getAlertStatus();
const history = await alerting.getAlertHistory(200);
const metrics = await alerting.getMetrics('gpu_memory_percent', 1000);

// Acknowledge alert to mark as acknowledged
await alerting.acknowledgeAlert(status.details[0].alert_id);

// Investigate root cause
analyzeMetricsTimeline(metrics);
```

## Best Practices

1. **Set realistic thresholds** based on baseline metrics
2. **Use appropriate severity levels** for context
3. **Configure hysteresis** to reduce flapping
4. **Monitor key metrics** consistently
5. **Acknowledge alerts** when investigating
6. **Use silence during maintenance** windows
7. **Archive old history** periodically
8. **Test alert conditions** before production
9. **Document thresholds** for team reference
10. **Monitor alert frequency** for drift

## Error Handling

```javascript
try {
  await alerting.registerAlert('Test', condition, 100);
} catch (error) {
  if (error.message.includes('severity')) {
    console.error('Invalid severity level');
  } else {
    console.error('Alert registration failed:', error);
  }
}
```

## Performance Notes

- Check cycles: ~5-50ms per cycle
- Metric storage: Last 1000 values per metric
- History: Last 1000 alert events
- Memory: ~10MB for 100 active alerts

## Integration Points

- **ModelVersioning**: Alert on model rollbacks
- **PatternDistribution**: Alert on sync failures
- **FederatedLearner**: Alert on consensus failures
- **SyncStrategy**: Alert on sync delays
- **Monitoring**: Feed into dashboards

## Notification Channels

### Log Channel

```
[SystemAlerting] ⚠ [critical] High GPU Memory: triggered
```

### Event Channel

Emits events for subscribers:
```javascript
on('alert:triggered', { alert_id, alert_name, severity })
on('alert:resolved', { alert_id })
```

### Webhook Channel

```javascript
POST https://alerts.example.com/webhook
{
  alert_id: 'alert-...',
  alert_name: 'High GPU Memory',
  severity: 'critical',
  event: 'triggered',
  value: 92,
  threshold: 85
}
```

## Limits and Defaults

| Parameter | Default | Max |
|-----------|---------|-----|
| Check interval | 10s | N/A |
| Hysteresis window | 3 | 10 |
| Max history | 1000 | 10000 |
| Metric retention | 1000 points | 10000 |
| Silence duration | 1h | Unlimited |

## Monitoring the Monitor

Key health checks:
- `checkCycles` increasing: System running
- `avgResponseTime` low: Good performance
- `alertsTriggered` vs `alertsResolved`: Ratio healthy
- `historySize` < `maxAlertHistory`: Not overflowing
