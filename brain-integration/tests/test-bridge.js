/**
 * BRAIN INTEGRATION BRIDGE - TESTS
 * 
 * Test suite for BrainIntegrationBridge
 * Covers: initialization, caching, error handling, metrics, feedback
 */

import { BrainIntegrationBridge } from '../brain-integration-bridge.js';

// Test utilities
const assert = (condition, message) => {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
};

const assertEquals = (actual, expected, message) => {
  if (actual !== expected) throw new Error(`${message}: expected ${expected}, got ${actual}`);
};

const assertArrayIncludes = (arr, item, message) => {
  if (!arr.includes(item)) throw new Error(`${message}: ${item} not in array`);
};

// ============================================================================
// Test Suite
// ============================================================================

let testsPassed = 0;
let testsFailed = 0;

async function runTests() {
  console.log('🧪 Starting Bridge Tests...\n');

  // Test 1: Initialization
  await test('Initialization', () => {
    const bridge = new BrainIntegrationBridge();
    assert(bridge !== null, 'Bridge created');
    assert(bridge.cache instanceof Map, 'Cache initialized');
    assert(bridge.metrics instanceof Map, 'Metrics initialized');
    assert(bridge.config.cacheSize === 1000, 'Default config applied');
  });

  // Test 2: Custom Configuration
  await test('Custom Configuration', () => {
    const bridge = new BrainIntegrationBridge({
      cacheSize: 500,
      defaultTTL: 1800000,
      maxRetries: 5
    });
    assertEquals(bridge.config.cacheSize, 500, 'Custom cache size');
    assertEquals(bridge.config.defaultTTL, 1800000, 'Custom TTL');
    assertEquals(bridge.config.maxRetries, 5, 'Custom retries');
  });

  // Test 3: Handler Registration
  await test('Handler Registration', () => {
    const bridge = new BrainIntegrationBridge();
    const mockHandler = async (data) => ({ result: 'test' });
    bridge.registerHandler('test-request', mockHandler);
    assert(bridge.handlers.has('test-request'), 'Handler registered');
  });

  // Test 4: Request Handling
  await test('Request Handling', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('echo', async (data) => ({ echo: data }));
    
    const response = await bridge.handle('echo', { message: 'hello' });
    assertEquals(response.echo.message, 'hello', 'Echo response');
  });

  // Test 5: Caching - Basic
  await test('Caching - Basic', async () => {
    const bridge = new BrainIntegrationBridge();
    let callCount = 0;
    
    bridge.registerHandler('counter', async (data) => {
      callCount++;
      return { count: callCount };
    });

    const resp1 = await bridge.handle('counter', { x: 1 });
    const resp2 = await bridge.handle('counter', { x: 1 }, { useCache: true });
    
    assertEquals(resp1.count, 1, 'First call');
    assertEquals(resp2.count, 1, 'Cached call returns same result');
    assertEquals(callCount, 1, 'Handler called once');
  });

  // Test 6: Caching - Different Data
  await test('Caching - Different Data', async () => {
    const bridge = new BrainIntegrationBridge();
    let callCount = 0;
    
    bridge.registerHandler('counter', async (data) => {
      callCount++;
      return { count: callCount, value: data.x };
    });

    const resp1 = await bridge.handle('counter', { x: 1 });
    const resp2 = await bridge.handle('counter', { x: 2 });
    
    assertEquals(callCount, 2, 'Different data triggers new call');
    assertEquals(resp1.value, 1, 'First response');
    assertEquals(resp2.value, 2, 'Second response');
  });

  // Test 7: Caching - Bypass
  await test('Caching - Bypass', async () => {
    const bridge = new BrainIntegrationBridge();
    let callCount = 0;
    
    bridge.registerHandler('counter', async () => {
      callCount++;
      return { count: callCount };
    });

    await bridge.handle('counter', { x: 1 });
    await bridge.handle('counter', { x: 1 }, { useCache: false });
    
    assertEquals(callCount, 2, 'Bypass cache on second call');
  });

  // Test 8: Cache Stats
  await test('Cache Stats', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('echo', async (data) => data);
    
    await bridge.handle('echo', { x: 1 });
    await bridge.handle('echo', { x: 1 });
    
    const stats = bridge.getCacheStats();
    assertEquals(stats.size, 1, 'Cache size is 1');
    assertEquals(stats.hits, 1, 'Cache hits is 1');
  });

  // Test 9: Error Handling
  await test('Error Handling', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('error-handler', async () => {
      throw new Error('Test error');
    });

    let caught = false;
    try {
      await bridge.handle('error-handler', {});
    } catch (error) {
      caught = true;
    }
    
    assert(caught, 'Error thrown');
  });

  // Test 10: Fallback Handler
  await test('Fallback Handler', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('failing', async () => {
      throw new Error('Test error');
    });

    const response = await bridge.handle('failing', {}, {
      fallback: (data, error) => ({ fallback: true, error: error.message })
    });

    assert(response.fallback === true, 'Fallback triggered');
    assertArrayIncludes(response.error, 'Test error', 'Error message included');
  });

  // Test 11: Metrics - Success
  await test('Metrics - Success', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('test', async () => ({ result: 'ok' }));
    
    await bridge.handle('test', {});
    const metrics = bridge.getMetrics('test');
    
    assertEquals(metrics.requests, 1, 'Request count');
    assertEquals(metrics.successes, 1, 'Success count');
    assertEquals(metrics.errors, 0, 'Error count');
  });

  // Test 12: Metrics - Error
  await test('Metrics - Error', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('failing', async () => {
      throw new Error('Test');
    });
    
    try {
      await bridge.handle('failing', {});
    } catch {}
    
    const metrics = bridge.getMetrics('failing');
    assertEquals(metrics.requests, 1, 'Request count');
    assertEquals(metrics.errors, 1, 'Error count');
  });

  // Test 13: Feedback Recording
  await test('Feedback Recording', async () => {
    const bridge = new BrainIntegrationBridge();
    
    bridge.recordFeedback(
      'test-type',
      { query: 'test' },
      { result: 'response' },
      { success: true, latency: 100 }
    );

    const feedback = bridge.getFeedback('test-type');
    assertEquals(feedback.length, 1, 'Feedback recorded');
    assertEquals(feedback[0].requestType, 'test-type', 'Feedback type');
  });

  // Test 14: Feedback Analysis
  await test('Feedback Analysis', async () => {
    const bridge = new BrainIntegrationBridge();
    
    for (let i = 0; i < 3; i++) {
      bridge.recordFeedback(
        'test',
        { x: i },
        { result: i },
        { success: i < 2, latency: 100 + i * 10 }
      );
    }

    const analysis = bridge.analyzeFeedback('test');
    assertEquals(analysis.count, 3, 'Feedback count');
    assertEquals(analysis.success, 2, 'Success count');
  });

  // Test 15: Runtime Registration
  await test('Runtime Registration', () => {
    const bridge = new BrainIntegrationBridge();
    const adapter = { id: 'test-runtime' };
    
    bridge.registerRuntime('test', adapter);
    assert(bridge.registered.has('test'), 'Runtime registered');
  });

  // Test 16: Runtime Unregistration
  await test('Runtime Unregistration', () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerRuntime('test', {});
    bridge.unregisterRuntime('test');
    
    assert(!bridge.registered.has('test'), 'Runtime unregistered');
  });

  // Test 17: Event Emission
  await test('Event Emission', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('test', async () => ({ result: 'ok' }));
    
    let eventFired = false;
    bridge.on('request-success', () => {
      eventFired = true;
    });

    await bridge.handle('test', {});
    assert(eventFired, 'Event emitted');
  });

  // Test 18: Cache Cleanup
  await test('Cache Cleanup', async () => {
    const bridge = new BrainIntegrationBridge({
      defaultTTL: 100 // Very short TTL
    });
    bridge.registerHandler('test', async () => ({ result: 'ok' }));

    await bridge.handle('test', { x: 1 });
    await new Promise(resolve => setTimeout(resolve, 150)); // Wait for expiration
    
    bridge.cleanupCache();
    const stats = bridge.getCacheStats();
    assertEquals(stats.expiredEntries, 0, 'Expired entries cleaned');
  });

  // Test 19: LRU Eviction
  await test('LRU Eviction', async () => {
    const bridge = new BrainIntegrationBridge({
      cacheSize: 2
    });
    bridge.registerHandler('echo', async (data) => data);

    await bridge.handle('echo', { x: 1 });
    await bridge.handle('echo', { x: 2 });
    await bridge.handle('echo', { x: 3 }); // Should evict x:1

    const stats = bridge.getCacheStats();
    assertEquals(stats.size, 2, 'Cache size maintained at limit');
  });

  // Test 20: Clear Cache
  await test('Clear Cache', async () => {
    const bridge = new BrainIntegrationBridge();
    bridge.registerHandler('test', async () => ({ result: 'ok' }));

    await bridge.handle('test', { x: 1 });
    await bridge.handle('test', { x: 2 });

    bridge.clearCache();
    const stats = bridge.getCacheStats();
    assertEquals(stats.size, 0, 'Cache cleared');
  });

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log(`Tests Passed: ${testsPassed}`);
  console.log(`Tests Failed: ${testsFailed}`);
  console.log(`Total: ${testsPassed + testsFailed}`);
  console.log('='.repeat(60));

  process.exit(testsFailed > 0 ? 1 : 0);
}

// Helper to run individual tests
async function test(name, fn) {
  try {
    await fn();
    console.log(`✓ ${name}`);
    testsPassed++;
  } catch (error) {
    console.error(`✗ ${name}`);
    console.error(`  ${error.message}`);
    testsFailed++;
  }
}

// Run tests
runTests().catch(error => {
  console.error('Fatal test error:', error);
  process.exit(1);
});
