/**
 * BRAIN INTEGRATION BRIDGE
 * 
 * Core coordinator for brain ↔ runtime system communication
 * Provides:
 * - Unified API for all runtime systems (MoS, XJSON, Micronaut, Scheduler)
 * - Event bus for cross-layer coordination
 * - Caching of brain results (TTL-based)
 * - Error handling & fallback logic
 * - Learning feedback collection
 * - Performance metrics & monitoring
 * 
 * @artifact brain/integration/bridge/v1
 */

import crypto from 'crypto';
import EventEmitter from 'events';

/**
 * Cache entry for brain results
 */
class CacheEntry {
  constructor(key, value, ttl = 3600000) { // 1 hour default
    this.key = key;
    this.value = value;
    this.ttl = ttl;
    this.timestamp = Date.now();
    this.hits = 0;
    this.accessed = Date.now();
  }

  isExpired() {
    return Date.now() - this.timestamp > this.ttl;
  }

  hit() {
    this.hits++;
    this.accessed = Date.now();
  }
}

/**
 * Main Integration Bridge
 */
export class BrainIntegrationBridge extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      cacheSize: config.cacheSize || 1000,
      defaultTTL: config.defaultTTL || 3600000, // 1 hour
      enableCaching: config.enableCaching !== false,
      enableMetrics: config.enableMetrics !== false,
      maxRetries: config.maxRetries || 3,
      retryDelay: config.retryDelay || 100,
      ...config
    };

    // Internal state
    this.cache = new Map();
    this.metrics = new Map();
    this.feedbackLog = [];
    this.registered = new Map(); // Registered runtime adapters
    this.handlers = new Map(); // Request handlers by type
    
    this.initialize();
  }

  initialize() {
    console.log('[BrainBridge] Initializing...');
    
    // Start cache cleanup interval
    this.cleanupInterval = setInterval(() => this.cleanupCache(), 60000); // Every minute
    
    // Initialize default handlers
    this.registerHandler('specialist-selection', this._handleSpecialistSelection.bind(this));
    this.registerHandler('program-intent', this._handleProgramIntent.bind(this));
    this.registerHandler('field-recommendation', this._handleFieldRecommendation.bind(this));
    this.registerHandler('task-priority', this._handleTaskPriority.bind(this));
    
    console.log('[BrainBridge] Initialized');
  }

  /**
   * Cleanup resources
   */
  destroy() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  // ============================================================================
  // Public API: Runtime Registration
  // ============================================================================

  /**
   * Register a runtime system adapter
   */
  registerRuntime(runtimeId, adapter) {
    if (this.registered.has(runtimeId)) {
      console.warn(`[BrainBridge] Runtime ${runtimeId} already registered, replacing`);
    }
    this.registered.set(runtimeId, adapter);
    console.log(`[BrainBridge] Registered runtime: ${runtimeId}`);
    this.emit('runtime-registered', { runtimeId, adapter });
  }

  /**
   * Unregister a runtime
   */
  unregisterRuntime(runtimeId) {
    this.registered.delete(runtimeId);
    console.log(`[BrainBridge] Unregistered runtime: ${runtimeId}`);
    this.emit('runtime-unregistered', { runtimeId });
  }

  // ============================================================================
  // Public API: Request Handling
  // ============================================================================

  /**
   * Handle a request from any runtime
   * 
   * @param {string} requestType - Type of request (specialist-selection, program-intent, etc)
   * @param {object} data - Request data
   * @param {object} options - Options (caching, retries, etc)
   * @returns {Promise<object>} Response from handler
   */
  async handle(requestType, data, options = {}) {
    const startTime = Date.now();
    const requestId = this._generateRequestId();
    
    try {
      console.log(`[BrainBridge:${requestId}] Request: ${requestType}`, data);

      // Check cache first
      if (this.config.enableCaching && options.useCache !== false) {
        const cached = this.getCached(requestType, data);
        if (cached) {
          const latency = Date.now() - startTime;
          this._recordMetric(requestType, 'cache-hit', latency);
          console.log(`[BrainBridge:${requestId}] Cache hit (${latency}ms)`);
          this.emit('request-cached', { requestId, requestType, latency });
          return cached;
        }
      }

      // Get handler
      const handler = this.handlers.get(requestType);
      if (!handler) {
        throw new Error(`No handler registered for request type: ${requestType}`);
      }

      // Execute handler with retry logic
      const response = await this._executeWithRetry(
        () => handler(data, options),
        options.retries ?? this.config.maxRetries,
        options.retryDelay ?? this.config.retryDelay
      );

      // Cache result
      if (this.config.enableCaching && options.cacheable !== false) {
        this.setCached(requestType, data, response, options.ttl);
      }

      const latency = Date.now() - startTime;
      this._recordMetric(requestType, 'success', latency);
      console.log(`[BrainBridge:${requestId}] Success (${latency}ms)`);
      this.emit('request-success', { requestId, requestType, latency, response });

      return response;

    } catch (error) {
      const latency = Date.now() - startTime;
      this._recordMetric(requestType, 'error', latency);
      console.error(`[BrainBridge:${requestId}] Error:`, error.message);
      this.emit('request-error', { requestId, requestType, latency, error });
      
      // Try fallback
      if (options.fallback) {
        console.log(`[BrainBridge:${requestId}] Attempting fallback...`);
        return options.fallback(data, error);
      }
      
      throw error;
    }
  }

  // ============================================================================
  // Public API: Handler Registration
  // ============================================================================

  /**
   * Register a custom handler for a request type
   */
  registerHandler(requestType, handler) {
    this.handlers.set(requestType, handler);
    console.log(`[BrainBridge] Registered handler: ${requestType}`);
  }

  // ============================================================================
  // Public API: Caching
  // ============================================================================

  /**
   * Get cached result
   */
  getCached(requestType, data) {
    if (!this.config.enableCaching) return null;

    const key = this._getCacheKey(requestType, data);
    const entry = this.cache.get(key);

    if (entry && !entry.isExpired()) {
      entry.hit();
      return entry.value;
    }

    if (entry && entry.isExpired()) {
      this.cache.delete(key);
    }

    return null;
  }

  /**
   * Set cache entry
   */
  setCached(requestType, data, value, ttl = null) {
    if (!this.config.enableCaching) return;

    const key = this._getCacheKey(requestType, data);
    const entry = new CacheEntry(key, value, ttl ?? this.config.defaultTTL);
    
    // Evict if cache is full
    if (this.cache.size >= this.config.cacheSize) {
      this._evictLRU();
    }

    this.cache.set(key, entry);
  }

  /**
   * Clear all cache
   */
  clearCache() {
    this.cache.clear();
    console.log('[BrainBridge] Cache cleared');
  }

  /**
   * Get cache stats
   */
  getCacheStats() {
    const entries = Array.from(this.cache.values());
    return {
      size: this.cache.size,
      maxSize: this.config.cacheSize,
      hits: entries.reduce((sum, e) => sum + e.hits, 0),
      totalEntries: entries.length,
      expiredEntries: entries.filter(e => e.isExpired()).length
    };
  }

  // ============================================================================
  // Public API: Learning Feedback
  // ============================================================================

  /**
   * Record feedback from runtime execution
   */
  recordFeedback(requestType, requestData, responseData, executionResult) {
    const feedback = {
      timestamp: Date.now(),
      requestType,
      requestData,
      responseData,
      executionResult,
      requestId: this._generateRequestId()
    };

    this.feedbackLog.push(feedback);
    console.log(`[BrainBridge] Feedback recorded: ${feedback.requestId}`);
    this.emit('feedback-recorded', feedback);

    // Keep only recent feedback (last 10000 entries)
    if (this.feedbackLog.length > 10000) {
      this.feedbackLog = this.feedbackLog.slice(-10000);
    }

    return feedback;
  }

  /**
   * Get feedback for a request type
   */
  getFeedback(requestType, limit = 100) {
    return this.feedbackLog
      .filter(f => f.requestType === requestType)
      .slice(-limit);
  }

  /**
   * Analyze feedback for a request type
   */
  analyzeFeedback(requestType) {
    const feedback = this.getFeedback(requestType, 1000);
    
    if (feedback.length === 0) {
      return { count: 0, success: 0, error: 0, avgLatency: 0 };
    }

    const successful = feedback.filter(f => f.executionResult?.success);
    const totalLatency = feedback.reduce((sum, f) => sum + (f.executionResult?.latency ?? 0), 0);

    return {
      count: feedback.length,
      success: successful.length,
      error: feedback.length - successful.length,
      successRate: successful.length / feedback.length,
      avgLatency: totalLatency / feedback.length,
      confidences: feedback.map(f => f.responseData?.confidence ?? 0)
    };
  }

  // ============================================================================
  // Public API: Metrics
  // ============================================================================

  /**
   * Get metrics for a request type
   */
  getMetrics(requestType = null) {
    if (requestType) {
      return this.metrics.get(requestType) || { requests: 0, successes: 0, errors: 0, avgLatency: 0 };
    }
    return Object.fromEntries(this.metrics);
  }

  /**
   * Clear metrics
   */
  clearMetrics() {
    this.metrics.clear();
    console.log('[BrainBridge] Metrics cleared');
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  /**
   * Generate unique request ID
   */
  _generateRequestId() {
    return `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate cache key from request type and data
   */
  _getCacheKey(requestType, data) {
    const hash = crypto
      .createHash('sha256')
      .update(JSON.stringify(data))
      .digest('hex')
      .substr(0, 8);
    return `${requestType}:${hash}`;
  }

  /**
   * Execute with retry logic
   */
  async _executeWithRetry(fn, maxRetries, delay) {
    if (maxRetries <= 0) return await fn();
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
      }
    }
  }

  /**
   * Record a metric
   */
  _recordMetric(requestType, status, latency) {
    if (!this.config.enableMetrics) return;

    const key = requestType;
    let metric = this.metrics.get(key) || {
      requests: 0,
      successes: 0,
      errors: 0,
      totalLatency: 0,
      latencies: []
    };

    metric.requests++;
    if (status === 'success' || status === 'cache-hit') {
      metric.successes++;
    } else if (status === 'error') {
      metric.errors++;
    }
    metric.totalLatency += latency;
    metric.latencies.push(latency);

    // Keep only recent latencies
    if (metric.latencies.length > 100) {
      metric.latencies = metric.latencies.slice(-100);
    }

    metric.avgLatency = metric.totalLatency / metric.requests;

    this.metrics.set(key, metric);
  }

  /**
   * Cleanup expired cache entries
   */
  cleanupCache() {
    let removed = 0;
    for (const [key, entry] of this.cache.entries()) {
      if (entry.isExpired()) {
        this.cache.delete(key);
        removed++;
      }
    }
    if (removed > 0) {
      console.log(`[BrainBridge] Cache cleanup: removed ${removed} expired entries`);
    }
  }

  /**
   * Evict least-recently-used entry
   */
  _evictLRU() {
    let lruKey = null;
    let lruTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.accessed < lruTime) {
        lruTime = entry.accessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
      console.log(`[BrainBridge] Evicted LRU cache entry: ${lruKey}`);
    }
  }

  // ============================================================================
  // Default Handlers (stubs - to be implemented)
  // ============================================================================

  async _handleSpecialistSelection(data, options) {
    // TODO: Call brain system for specialist ranking
    return {
      ranking: [],
      confidence: 0.5,
      reasoning: 'Not yet implemented'
    };
  }

  async _handleProgramIntent(data, options) {
    // TODO: Analyze program intent
    return {
      intent: null,
      confidence: 0.5,
      suggestions: []
    };
  }

  async _handleFieldRecommendation(data, options) {
    // TODO: Recommend field configuration
    return {
      fields: [],
      confidence: 0.5,
      alternatives: []
    };
  }

  async _handleTaskPriority(data, options) {
    // TODO: Determine task priority
    return {
      priority: 0.5,
      confidence: 0.5,
      reasoning: ''
    };
  }
}

export default BrainIntegrationBridge;
