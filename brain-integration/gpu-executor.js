/**
 * GPU Executor Layer
 * 
 * Routes computational operations to GPU or CPU based on:
 * - GPU availability (from GPUDetector)
 * - Operation type and data size
 * - Historical performance metrics
 */

import { GPUDetector } from './gpu-detection.js';

class GPUExecutor {
  constructor(gpuDetector = null, metricsStorage = null) {
    this.gpuDetector = gpuDetector;
    this.metricsStorage = metricsStorage;
    this.operationRegistry = new Map();
    this.performanceCache = new Map();
    this.initialized = false;

    // Registered GPU operations
    this._registerDefaultOperations();
  }

  /**
   * Initialize GPU executor
   */
  async initialize() {
    console.log('🚀 GPU Executor: Initializing...');

    if (!this.gpuDetector) {
      this.gpuDetector = new GPUDetector();
      await this.gpuDetector.initialize();
    }

    this.initialized = true;
    console.log('✓ GPU Executor initialized');

    return {
      initialized: true,
      gpu_available: this.gpuDetector.shouldUseGPU(),
      capabilities: this.gpuDetector.getGPUInfo().capabilities,
      registered_operations: this.operationRegistry.size,
    };
  }

  /**
   * Register a GPU operation
   * @param {string} operationName - Name of operation (e.g., 'matrix-multiply')
   * @param {function} gpuImpl - GPU implementation (async)
   * @param {function} cpuImpl - CPU implementation (async)
   * @param {object} config - Operation configuration
   */
  registerOperation(operationName, gpuImpl, cpuImpl, config = {}) {
    this.operationRegistry.set(operationName, {
      gpuImpl,
      cpuImpl,
      config: {
        gpu_threshold_kb: config.gpu_threshold_kb || 1024, // Min data size for GPU
        gpu_min_time_ms: config.gpu_min_time_ms || 5, // Min time before GPU is worth it
        gpu_overhead_ms: config.gpu_overhead_ms || 2, // Transfer overhead
        prefer_gpu: config.prefer_gpu !== false, // Default to GPU if available
        ...config,
      },
    });
  }

  /**
   * Execute operation (routes to GPU or CPU)
   * @param {string} operationName - Registered operation name
   * @param {object} data - Input data
   * @param {object} options - Execution options
   * @returns {Promise} Operation result
   */
  async execute(operationName, data, options = {}) {
    if (!this.initialized) {
      throw new Error('GPU Executor not initialized');
    }

    const operation = this.operationRegistry.get(operationName);
    if (!operation) {
      throw new Error(`Unknown operation: ${operationName}`);
    }

    const startTime = Date.now();
    const dataSize = this._estimateDataSize(data);

    // Decide routing
    const useGPU = this._decideGPURoute(operationName, data, operation, options);

    try {
      let result;

      if (useGPU) {
        console.log(`⚡ Executing ${operationName} on GPU`);
        result = await operation.gpuImpl(data);
      } else {
        console.log(`💻 Executing ${operationName} on CPU`);
        result = await operation.cpuImpl(data);
      }

      const duration = Date.now() - startTime;

      // Record metrics
      if (this.metricsStorage) {
        this.metricsStorage.recordGPUComparison({
          operation: operationName,
          used_gpu: useGPU,
          execution_time: duration,
          data_size_kb: dataSize,
          success: true,
        });
      }

      return {
        result,
        metadata: {
          executed_on: useGPU ? 'GPU' : 'CPU',
          duration_ms: duration,
          data_size_kb: dataSize,
          timestamp: new Date().toISOString(),
        },
      };
    } catch (error) {
      console.error(`❌ Execution failed for ${operationName}:`, error.message);

      // Fallback to CPU on GPU error
      if (useGPU) {
        console.log(`⚠️  GPU failed, falling back to CPU`);
        const cpuStartTime = Date.now();

        try {
          const result = await operation.cpuImpl(data);
          const duration = Date.now() - cpuStartTime;

          if (this.metricsStorage) {
            this.metricsStorage.recordGPUComparison({
              operation: operationName,
              used_gpu: false,
              execution_time: duration,
              data_size_kb: dataSize,
              success: true,
              fallback_reason: 'GPU error',
            });
          }

          return {
            result,
            metadata: {
              executed_on: 'CPU (fallback)',
              duration_ms: duration,
              data_size_kb: dataSize,
              fallback_reason: error.message,
              timestamp: new Date().toISOString(),
            },
          };
        } catch (cpuError) {
          throw new Error(`Both GPU and CPU execution failed: ${cpuError.message}`);
        }
      }

      throw error;
    }
  }

  /**
   * Decide whether to use GPU or CPU
   * @private
   */
  _decideGPURoute(operationName, data, operation, options) {
    // Explicit override
    if (options.forceGPU === true) return true;
    if (options.forceCPU === true) return false;

    // GPU not available
    if (!this.gpuDetector.shouldUseGPU()) {
      return false;
    }

    const config = operation.config;
    const dataSize = this._estimateDataSize(data);

    // Data too small for GPU overhead
    if (dataSize < config.gpu_threshold_kb) {
      return false;
    }

    // Check historical performance
    const performanceRatio = this._getPerformanceRatio(operationName);
    if (performanceRatio !== null && performanceRatio < 0.9) {
      // CPU is > 10% faster historically
      return false;
    }

    // Default decision
    return config.prefer_gpu && this.gpuDetector.hasCapability('cuda');
  }

  /**
   * Estimate data size in KB
   * @private
   */
  _estimateDataSize(data) {
    if (typeof data === 'object' && data !== null) {
      if (Array.isArray(data)) {
        // Estimate: 8 bytes per number
        return (data.length * 8) / 1024;
      }

      if (data.buffer instanceof ArrayBuffer) {
        return data.buffer.byteLength / 1024;
      }

      // JSON stringify estimate
      const jsonSize = JSON.stringify(data).length;
      return jsonSize / 1024;
    }

    return 0;
  }

  /**
   * Get historical GPU vs CPU performance ratio
   * @private
   */
  _getPerformanceRatio(operationName) {
    const cached = this.performanceCache.get(operationName);
    if (cached) {
      return cached.ratio;
    }

    // TODO: Query metrics storage for historical performance
    // gpuTime vs cpuTime average

    return null; // No historical data
  }

  /**
   * Benchmark operation (GPU vs CPU)
   * @param {string} operationName - Operation to benchmark
   * @param {object} benchmarkData - Test data for benchmark
   * @param {object} options - Benchmark options
   * @returns {Promise} Benchmark results
   */
  async benchmark(operationName, benchmarkData, options = {}) {
    const operation = this.operationRegistry.get(operationName);
    if (!operation) {
      throw new Error(`Unknown operation: ${operationName}`);
    }

    const iterations = options.iterations || 5;
    const gpuTimes = [];
    const cpuTimes = [];

    console.log(`📊 Benchmarking ${operationName} (${iterations} iterations)`);

    // Benchmark GPU
    if (this.gpuDetector.shouldUseGPU()) {
      console.log('  Testing GPU...');

      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();

        try {
          await operation.gpuImpl(benchmarkData);
          gpuTimes.push(Date.now() - startTime);
        } catch (error) {
          console.warn(`  GPU iteration ${i + 1} failed:`, error.message);
        }
      }
    }

    // Benchmark CPU
    console.log('  Testing CPU...');

    for (let i = 0; i < iterations; i++) {
      const startTime = Date.now();

      try {
        await operation.cpuImpl(benchmarkData);
        cpuTimes.push(Date.now() - startTime);
      } catch (error) {
        console.warn(`  CPU iteration ${i + 1} failed:`, error.message);
      }
    }

    // Analyze results
    const gpuAvg = gpuTimes.length > 0 ? gpuTimes.reduce((a, b) => a + b) / gpuTimes.length : null;
    const cpuAvg = cpuTimes.length > 0 ? cpuTimes.reduce((a, b) => a + b) / cpuTimes.length : null;
    const speedup = (gpuAvg !== null && cpuAvg !== null && gpuAvg > 0) ? cpuAvg / gpuAvg : null;

    const results = {
      operation: operationName,
      iterations,
      gpu: {
        available: gpuTimes.length > 0,
        times: gpuTimes,
        average_ms: gpuAvg,
        min_ms: gpuTimes.length > 0 ? Math.min(...gpuTimes) : null,
        max_ms: gpuTimes.length > 0 ? Math.max(...gpuTimes) : null,
      },
      cpu: {
        times: cpuTimes,
        average_ms: cpuAvg,
        min_ms: Math.min(...cpuTimes),
        max_ms: Math.max(...cpuTimes),
      },
      analysis: {
        speedup_ratio: speedup,
        gpu_faster: speedup !== null && speedup > 1.0,
        recommendation: speedup !== null && speedup > 1.2 ? 'use-gpu' : 'use-cpu',
      },
      timestamp: new Date().toISOString(),
    };

    // Cache result
    if (speedup !== null) {
      this.performanceCache.set(operationName, {
        ratio: 1 / speedup, // GPU time / CPU time
        timestamp: Date.now(),
      });
    }

    console.log(`  Results: GPU ${results.gpu.average_ms !== null ? results.gpu.average_ms + 'ms' : 'N/A'} vs CPU ${results.cpu.average_ms}ms`);
    if (speedup) {
      console.log(`  Speedup: ${speedup.toFixed(2)}x`);
    }

    return results;
  }

  /**
   * Get executor status and capabilities
   */
  getStatus() {
    return {
      initialized: this.initialized,
      gpu_available: this.gpuDetector?.shouldUseGPU() || false,
      gpu_device_count: this.gpuDetector?.getGPUInfo().devices.length || 0,
      gpu_memory_mb: this.gpuDetector?.getTotalMemory() || 0,
      registered_operations: this.operationRegistry.size,
      cached_benchmarks: this.performanceCache.size,
      capabilities: this.gpuDetector?.getGPUInfo().capabilities || {},
    };
  }

  /**
   * Register default tensor operations
   * @private
   */
  _registerDefaultOperations() {
    // Matrix Multiply
    this.registerOperation(
      'matrix-multiply',
      async (data) => {
        // GPU implementation: would use CUDA/GPU backend
        return this._gpuMatrixMultiply(data);
      },
      async (data) => {
        // CPU implementation: fallback
        return this._cpuMatrixMultiply(data);
      },
      {
        gpu_threshold_kb: 512,
        gpu_min_time_ms: 10,
        prefer_gpu: true,
      }
    );

    // FFT (Fast Fourier Transform)
    this.registerOperation(
      'fft',
      async (data) => {
        return this._gpuFFT(data);
      },
      async (data) => {
        return this._cpuFFT(data);
      },
      {
        gpu_threshold_kb: 256,
        gpu_min_time_ms: 5,
        prefer_gpu: true,
      }
    );

    // Convolution
    this.registerOperation(
      'convolution',
      async (data) => {
        return this._gpuConvolution(data);
      },
      async (data) => {
        return this._cpuConvolution(data);
      },
      {
        gpu_threshold_kb: 1024,
        gpu_min_time_ms: 15,
        prefer_gpu: true,
      }
    );

    // Reduction (sum, max, min)
    this.registerOperation(
      'reduction',
      async (data) => {
        return this._gpuReduction(data);
      },
      async (data) => {
        return this._cpuReduction(data);
      },
      {
        gpu_threshold_kb: 128,
        gpu_min_time_ms: 2,
        prefer_gpu: true,
      }
    );

    // Sorting
    this.registerOperation(
      'sort',
      async (data) => {
        return this._gpuSort(data);
      },
      async (data) => {
        return this._cpuSort(data);
      },
      {
        gpu_threshold_kb: 2048,
        gpu_min_time_ms: 20,
        prefer_gpu: false, // CPU often faster for sort
      }
    );

    // Transpose
    this.registerOperation(
      'transpose',
      async (data) => {
        return this._gpuTranspose(data);
      },
      async (data) => {
        return this._cpuTranspose(data);
      },
      {
        gpu_threshold_kb: 512,
        gpu_min_time_ms: 5,
        prefer_gpu: true,
      }
    );
  }

  /**
   * Stub implementations (would use actual GPU libraries)
   * @private
   */

  async _gpuMatrixMultiply(data) {
    // Would use CUDA/GPU backend
    // Example: cudnn.matmul(data)
    await new Promise((r) => setTimeout(r, Math.random() * 10 + 5));
    return { result: 'gpu-matrix-multiply' };
  }

  async _cpuMatrixMultiply(data) {
    // CPU implementation
    await new Promise((r) => setTimeout(r, Math.random() * 50 + 20));
    return { result: 'cpu-matrix-multiply' };
  }

  async _gpuFFT(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 8 + 3));
    return { result: 'gpu-fft' };
  }

  async _cpuFFT(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 40 + 15));
    return { result: 'cpu-fft' };
  }

  async _gpuConvolution(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 12 + 6));
    return { result: 'gpu-convolution' };
  }

  async _cpuConvolution(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 100 + 40));
    return { result: 'cpu-convolution' };
  }

  async _gpuReduction(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 3 + 1));
    return { result: 'gpu-reduction' };
  }

  async _cpuReduction(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 20 + 5));
    return { result: 'cpu-reduction' };
  }

  async _gpuSort(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 20 + 10));
    return { result: 'gpu-sort' };
  }

  async _cpuSort(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 15 + 5));
    return { result: 'cpu-sort' };
  }

  async _gpuTranspose(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 5 + 2));
    return { result: 'gpu-transpose' };
  }

  async _cpuTranspose(data) {
    await new Promise((r) => setTimeout(r, Math.random() * 30 + 10));
    return { result: 'cpu-transpose' };
  }
}

export { GPUExecutor };
