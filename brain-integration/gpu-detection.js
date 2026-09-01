/**
 * GPU Detection Layer
 * 
 * Detects available GPU hardware and capabilities:
 * - NVIDIA CUDA (nvidia-smi)
 * - AMD ROCm (rocm-smi)
 * - Apple Metal (system introspection)
 * - WebGL (browser-based GPU)
 * - Intel GPU (Windows/Linux)
 */

import { execSync, exec } from 'child_process';
import os from 'os';
import path from 'path';

class GPUDetector {
  constructor() {
    this.gpuInfo = null;
    this.capabilities = {};
    this.initialized = false;
    this.detectionTime = null;
  }

  /**
   * Initialize GPU detection
   * Runs all detection methods in parallel, builds comprehensive GPU profile
   */
  async initialize() {
    console.log('🔍 GPU Detection: Starting...');
    const startTime = Date.now();

    try {
      // Run all detection methods in parallel
      const [nvidia, amd, metal, intel] = await Promise.all([
        this._detectNvidia(),
        this._detectAMD(),
        this._detectMetal(),
        this._detectIntel(),
      ]);

      this.gpuInfo = {
        timestamp: new Date().toISOString(),
        platform: os.platform(),
        arch: os.arch(),
        devices: [],
        capabilities: {
          cuda: false,
          rocm: false,
          metal: false,
          webgl: false,
          opencl: false,
          vulkan: false,
        },
      };

      // Add NVIDIA if detected
      if (nvidia) {
        this.gpuInfo.devices.push(nvidia);
        this.gpuInfo.capabilities.cuda = true;
      }

      // Add AMD if detected
      if (amd) {
        this.gpuInfo.devices.push(amd);
        this.gpuInfo.capabilities.rocm = true;
      }

      // Add Metal if detected
      if (metal) {
        this.gpuInfo.devices.push(metal);
        this.gpuInfo.capabilities.metal = true;
      }

      // Add Intel if detected
      if (intel) {
        this.gpuInfo.devices.push(intel);
        this.gpuInfo.capabilities.opencl = true;
      }

      this.detectionTime = Date.now() - startTime;
      this.initialized = true;

      this._logDetectionSummary();
      return this.gpuInfo;
    } catch (error) {
      console.error('❌ GPU Detection failed:', error.message);
      this.gpuInfo = this._getNoGPUProfile();
      this.initialized = true;
      return this.gpuInfo;
    }
  }

  /**
   * Detect NVIDIA GPU (CUDA)
   */
  async _detectNvidia() {
    try {
      const output = execSync('nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader', {
        encoding: 'utf8',
        timeout: 5000,
        stdio: ['pipe', 'pipe', 'pipe'],
      }).trim();

      if (!output) return null;

      const lines = output.split('\n');
      const devices = [];

      for (let i = 0; i < lines.length; i++) {
        const [index, name, driverVersion, memory] = lines[i].split(',').map((s) => s.trim());

        devices.push({
          type: 'NVIDIA',
          index: parseInt(index),
          name: name.trim(),
          driver_version: driverVersion.trim(),
          memory_total: this._parseMemory(memory),
          compute_capability: await this._getNvidiaComputeCapability(index),
          supports_cuda: true,
          supports_cudnn: true,
          supports_tensorrt: true,
        });
      }

      return {
        type: 'NVIDIA',
        count: devices.length,
        devices,
        capabilities: ['CUDA', 'cuDNN', 'TensorRT'],
        available: true,
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Get NVIDIA compute capability
   */
  async _getNvidiaComputeCapability(deviceIndex) {
    try {
      const output = execSync(
        `nvidia-smi -i ${deviceIndex} --query-gpu=compute_cap --format=csv,noheader`,
        {
          encoding: 'utf8',
          timeout: 3000,
        },
      ).trim();

      return output;
    } catch {
      return 'unknown';
    }
  }

  /**
   * Detect AMD GPU (ROCm)
   */
  async _detectAMD() {
    try {
      const output = execSync('rocm-smi --showid --showtemp --showram', {
        encoding: 'utf8',
        timeout: 5000,
        stdio: ['pipe', 'pipe', 'pipe'],
      }).trim();

      if (!output) return null;

      const devices = [];
      const lines = output.split('\n').filter((l) => l.includes('GPU'));

      for (let i = 0; i < lines.length; i++) {
        devices.push({
          type: 'AMD',
          index: i,
          name: `AMD GPU ${i}`,
          supports_rocm: true,
          supports_miopen: true,
        });
      }

      return {
        type: 'AMD',
        count: devices.length,
        devices,
        capabilities: ['ROCm', 'MIOpen'],
        available: true,
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Detect Apple Metal GPU
   */
  async _detectMetal() {
    try {
      if (os.platform() !== 'darwin') return null;

      // Apple Metal is integrated on all modern Macs
      const output = execSync('system_profiler SPDisplaysDataType', {
        encoding: 'utf8',
        timeout: 5000,
      }).trim();

      if (output.includes('Metal')) {
        return {
          type: 'Apple Metal',
          name: 'Integrated Metal GPU',
          supports_metal: true,
          supports_metal_performance_shaders: true,
          available: true,
          devices: [
            {
              type: 'Metal',
              name: 'System GPU',
              index: 0,
            },
          ],
        };
      }

      return null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Detect Intel GPU
   */
  async _detectIntel() {
    try {
      const platform = os.platform();

      if (platform === 'win32') {
        // Windows: Check DirectX 12 GPU capabilities
        try {
          const output = execSync('wmic path win32_videocontroller get name', {
            encoding: 'utf8',
            timeout: 5000,
          });

          const gpuLines = output
            .split('\n')
            .filter((l) => l && !l.includes('Name'))
            .map((l) => l.trim());

          if (gpuLines.some((l) => l.includes('Intel'))) {
            return {
              type: 'Intel',
              capabilities: ['OpenCL', 'DirectX12', 'OpenGL'],
              devices: gpuLines
                .filter((l) => l.includes('Intel'))
                .map((name, i) => ({
                  type: 'Intel GPU',
                  index: i,
                  name,
                })),
              available: true,
            };
          }
        } catch {
          return null;
        }
      }

      return null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Parse memory string (e.g., "11019 MiB" -> 11019)
   */
  _parseMemory(memStr) {
    const match = memStr.match(/(\d+)\s*(MiB|GiB|KiB|B)/i);
    if (!match) return 0;

    const [, value, unit] = match;
    const numValue = parseInt(value);

    switch (unit.toUpperCase()) {
      case 'GIB':
        return numValue * 1024;
      case 'MIB':
        return numValue;
      case 'KIB':
        return numValue / 1024;
      case 'B':
        return numValue / 1024 / 1024;
      default:
        return numValue;
    }
  }

  /**
   * Get GPU profile when no GPU found
   */
  _getNoGPUProfile() {
    return {
      timestamp: new Date().toISOString(),
      platform: os.platform(),
      arch: os.arch(),
      devices: [],
      capabilities: {
        cuda: false,
        rocm: false,
        metal: false,
        webgl: false,
        opencl: false,
        vulkan: false,
      },
    };
  }

  /**
   * Log detection summary
   */
  _logDetectionSummary() {
    console.log('\n📊 GPU Detection Summary:');
    console.log(`   Platform: ${this.gpuInfo.platform} (${this.gpuInfo.arch})`);
    console.log(`   Devices found: ${this.gpuInfo.devices.length}`);

    if (this.gpuInfo.devices.length === 0) {
      console.log('   ⚠️  No GPU detected - will use CPU fallback');
    } else {
      this.gpuInfo.devices.forEach((device) => {
        console.log(`   • ${device.type}: ${device.name || 'Unknown'}`);
        if (device.memory_total) {
          console.log(`     Memory: ${device.memory_total} MB`);
        }
        if (device.capabilities) {
          console.log(`     Capabilities: ${device.capabilities.join(', ')}`);
        }
      });
    }

    const caps = Object.entries(this.gpuInfo.capabilities)
      .filter(([_, v]) => v)
      .map(([k]) => k.toUpperCase());

    if (caps.length > 0) {
      console.log(`   ✓ Enabled: ${caps.join(', ')}`);
    } else {
      console.log('   ℹ️  No hardware acceleration available');
    }

    console.log(`   Detection time: ${this.detectionTime}ms\n`);
  }

  /**
   * Get GPU info (must call initialize first)
   */
  getGPUInfo() {
    if (!this.initialized) {
      throw new Error('GPU Detector not initialized. Call initialize() first.');
    }
    return this.gpuInfo;
  }

  /**
   * Check if specific capability is available
   */
  hasCapability(capability) {
    if (!this.initialized) return false;
    return this.gpuInfo.capabilities[capability.toLowerCase()] === true;
  }

  /**
   * Get total GPU memory across all devices
   */
  getTotalMemory() {
    if (!this.initialized) return 0;

    return this.gpuInfo.devices.reduce((sum, device) => {
      return sum + (device.memory_total || 0);
    }, 0);
  }

  /**
   * Get device by type
   */
  getDevicesByType(type) {
    if (!this.initialized) return [];

    return this.gpuInfo.devices.filter((d) => d.type.toUpperCase() === type.toUpperCase());
  }

  /**
   * Get first available GPU
   */
  getPrimaryGPU() {
    if (!this.initialized || this.gpuInfo.devices.length === 0) {
      return null;
    }

    return this.gpuInfo.devices[0];
  }

  /**
   * Check if GPU should be used for operation
   * Simple heuristic: use GPU if available and has >1GB memory
   */
  shouldUseGPU() {
    if (!this.initialized || this.gpuInfo.devices.length === 0) {
      return false;
    }

    const totalMemory = this.getTotalMemory();
    return totalMemory > 1024; // >1GB
  }

  /**
   * Get GPU statistics for logging/monitoring
   */
  getStats() {
    return {
      initialized: this.initialized,
      detection_time_ms: this.detectionTime,
      device_count: this.gpuInfo?.devices?.length || 0,
      total_memory_mb: this.getTotalMemory(),
      primary_gpu: this.getPrimaryGPU()?.name || 'None',
      capabilities_enabled: Object.entries(this.gpuInfo?.capabilities || {})
        .filter(([_, v]) => v)
        .map(([k]) => k),
      should_use_gpu: this.shouldUseGPU(),
    };
  }
}

export { GPUDetector };
