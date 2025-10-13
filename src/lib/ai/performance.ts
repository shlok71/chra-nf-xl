/**
 * Enhanced Performance Optimization Service
 * Optimizes response time with caching, streaming, and batching
 */

import { ENHANCED_MODEL_CONFIG, PERFORMANCE_SETTINGS } from './index';

export interface PerformanceMetrics {
  responseTime: number;
  cacheHitRate: number;
  throughput: number;
  memoryUsage: number;
  errorRate: number;
}

export interface CacheEntry {
  id: string;
  content: string;
  timestamp: number;
  ttl: number;
  hitCount: number;
  size: number;
}

export interface BatchRequest {
  id: string;
  requests: Array<{
    id: string;
    content: string;
    priority: number;
  }>;
  timestamp: number;
  status: 'pending' | 'processing' | 'completed';
}

class PerformanceOptimizer {
  private cache: Map<string, CacheEntry> = new Map();
  private batchQueue: BatchRequest[] = [];
  private metrics: PerformanceMetrics = {
    responseTime: 0,
    cacheHitRate: 0,
    throughput: 0,
    memoryUsage: 0,
    errorRate: 0
  };
  private processingBatch = false;
  private cacheStats = {
    hits: 0,
    misses: 0,
    totalRequests: 0
  };

  constructor() {
    // Initialize cache cleanup interval
    setInterval(() => this.cleanupCache(), 60000); // Clean up every minute
    
    // Initialize batch processing
    setInterval(() => this.processBatchQueue(), 100); // Process every 100ms
    
    // Initialize metrics collection
    setInterval(() => this.updateMetrics(), 5000); // Update every 5 seconds
  }

  /**
   * Optimized chat completion with caching and streaming
   */
  async optimizedChatCompletion(params: {
    messages: Array<{ role: string; content: string }>;
    stream?: boolean;
    priority?: number;
  }): Promise<any> {
    const startTime = Date.now();
    this.cacheStats.totalRequests++;

    // Generate cache key
    const cacheKey = this.generateCacheKey(params.messages);
    
    // Check cache first
    const cachedResponse = this.getFromCache(cacheKey);
    if (cachedResponse) {
      this.cacheStats.hits++;
      this.updateResponseTime(Date.now() - startTime);
      return cachedResponse;
    }

    this.cacheStats.misses++;

    // Apply compression if content is large
    const optimizedParams = this.optimizeRequest(params);
    
    try {
      // Use ZAI SDK with enhanced configuration
      const ZAI = await import('z-ai-web-dev-sdk');
      const zai = await ZAI.create();

      const completion = await zai.chat.completions.create({
        messages: optimizedParams.messages,
        max_tokens: ENHANCED_MODEL_CONFIG.maxOutputTokens,
        temperature: ENHANCED_MODEL_CONFIG.temperature,
        top_p: ENHANCED_MODEL_CONFIG.topP,
        stream: params.stream || ENHANCED_MODEL_CONFIG.responseOptimization.streaming,
        // Additional optimization parameters
        cache: ENHANCED_MODEL_CONFIG.responseOptimization.caching,
        compress: ENHANCED_MODEL_CONFIG.responseOptimization.compression,
        timeout: ENHANCED_MODEL_CONFIG.responseOptimization.timeoutMs
      });

      const responseTime = Date.now() - startTime;
      this.updateResponseTime(responseTime);

      // Cache the response
      if (completion.choices && completion.choices[0]?.message?.content) {
        this.setCache(cacheKey, completion.choices[0].message.content);
      }

      return completion;

    } catch (error) {
      console.error('Optimized chat completion failed:', error);
      this.metrics.errorRate = (this.metrics.errorRate + 1) / this.cacheStats.totalRequests;
      throw error;
    }
  }

  /**
   * Batch processing for multiple requests
   */
  async addToBatch(request: {
    id: string;
    content: string;
    priority?: number;
  }): Promise<any> {
    return new Promise((resolve, reject) => {
      const batchRequest: BatchRequest = {
        id: request.id,
        requests: [{
          id: request.id,
          content: request.content,
          priority: request.priority || 1
        }],
        timestamp: Date.now(),
        status: 'pending'
      };

      this.batchQueue.push(batchRequest);

      // Set up response handler
      const checkResponse = () => {
        const completed = this.batchQueue.find(br => br.id === request.id && br.status === 'completed');
        if (completed) {
          resolve(completed);
        } else {
          setTimeout(checkResponse, 50);
        }
      };

      checkResponse();
    });
  }

  /**
   * Process batch queue
   */
  private async processBatchQueue(): Promise<void> {
    if (this.processingBatch || this.batchQueue.length === 0) {
      return;
    }

    this.processingBatch = true;

    try {
      // Get batch of requests
      const batchSize = Math.min(
        ENHANCED_MODEL_CONFIG.responseOptimization.maxBatchSize,
        this.batchQueue.length
      );

      const batch = this.batchQueue.splice(0, batchSize);
      
      // Sort by priority
      batch.sort((a, b) => {
        const aPriority = a.requests[0]?.priority || 1;
        const bPriority = b.requests[0]?.priority || 1;
        return bPriority - aPriority;
      });

      // Process batch
      await this.processBatch(batch);

    } catch (error) {
      console.error('Batch processing failed:', error);
    } finally {
      this.processingBatch = false;
    }
  }

  /**
   * Process a batch of requests
   */
  private async processBatch(batch: BatchRequest[]): Promise<void> {
    const ZAI = await import('z-ai-web-dev-sdk');
    const zai = await ZAI.create();

    // Mark as processing
    batch.forEach(br => br.status = 'processing');

    try {
      // Process all requests in parallel
      const promises = batch.map(async (batchRequest) => {
        const request = batchRequest.requests[0];
        
        const completion = await zai.chat.completions.create({
          messages: [{ role: 'user', content: request.content }],
          max_tokens: ENHANCED_MODEL_CONFIG.maxOutputTokens,
          temperature: ENHANCED_MODEL_CONFIG.temperature,
          timeout: ENHANCED_MODEL_CONFIG.responseOptimization.timeoutMs
        });

        batchRequest.status = 'completed';
        return completion;
      });

      await Promise.all(promises);

    } catch (error) {
      // Mark failed requests
      batch.forEach(br => {
        if (br.status === 'processing') {
          br.status = 'pending'; // Retry later
        }
      });
    }
  }

  /**
   * Cache management
   */
  private generateCacheKey(messages: Array<{ role: string; content: string }>): string {
    const content = messages.map(m => `${m.role}:${m.content}`).join('|');
    return btoa(content).substring(0, 32);
  }

  private getFromCache(key: string): any | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }

    // Check TTL
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    // Update hit count
    entry.hitCount++;
    return JSON.parse(entry.content);
  }

  private setCache(key: string, content: any): void {
    const entry: CacheEntry = {
      id: key,
      content: JSON.stringify(content),
      timestamp: Date.now(),
      ttl: 300000, // 5 minutes TTL
      hitCount: 0,
      size: JSON.stringify(content).length
    };

    this.cache.set(key, entry);
  }

  private cleanupCache(): void {
    const now = Date.now();
    
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.cache.delete(key);
      }
    }

    // If cache is too large, remove least recently used entries
    if (this.cache.size > PERFORMANCE_SETTINGS.memory.cacheSize) {
      const entries = Array.from(this.cache.entries());
      entries.sort((a, b) => a[1].hitCount - b[1].hitCount);
      
      const toRemove = entries.slice(0, this.cache.size - PERFORMANCE_SETTINGS.memory.cacheSize);
      toRemove.forEach(([key]) => this.cache.delete(key));
    }
  }

  /**
   * Request optimization
   */
  private optimizeRequest(params: {
    messages: Array<{ role: string; content: string }>;
  }): typeof params {
    // Apply compression if content is large
    const optimizedMessages = params.messages.map(msg => {
      if (msg.content.length > PERFORMANCE_SETTINGS.memory.compressionThreshold) {
        // Compress long messages
        return {
          ...msg,
          content: this.compressContent(msg.content)
        };
      }
      return msg;
    });

    return {
      ...params,
      messages: optimizedMessages
    };
  }

  private compressContent(content: string): string {
    // Simple compression - remove extra whitespace and normalize
    return content
      .replace(/\s+/g, ' ')
      .replace(/\n+/g, ' ')
      .trim();
  }

  /**
   * Metrics management
   */
  private updateResponseTime(time: number): void {
    this.metrics.responseTime = (this.metrics.responseTime * 0.9) + (time * 0.1);
  }

  private updateMetrics(): void {
    // Update cache hit rate
    if (this.cacheStats.totalRequests > 0) {
      this.metrics.cacheHitRate = this.cacheStats.hits / this.cacheStats.totalRequests;
    }

    // Update memory usage
    this.metrics.memoryUsage = this.cache.size;

    // Update throughput (requests per second)
    this.metrics.throughput = this.cacheStats.totalRequests / 60; // Rough estimate
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return {
      ...this.cacheStats,
      cacheSize: this.cache.size,
      cacheEntries: Array.from(this.cache.values()).map(entry => ({
        id: entry.id,
        hitCount: entry.hitCount,
        size: entry.size,
        age: Date.now() - entry.timestamp
      }))
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    this.cacheStats = {
      hits: 0,
      misses: 0,
      totalRequests: 0
    };
  }

  /**
   * Preload common responses
   */
  async preloadCache(): Promise<void> {
    const commonPrompts = [
      "Hello, how are you?",
      "What can you help me with?",
      "Explain machine learning",
      "Help me write code",
      "Translate this text"
    ];

    const ZAI = await import('z-ai-web-dev-sdk');
    const zai = await ZAI.create();

    for (const prompt of commonPrompts) {
      try {
        const completion = await zai.chat.completions.create({
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 1000,
          temperature: 0.7
        });

        const cacheKey = this.generateCacheKey([{ role: 'user', content: prompt }]);
        this.setCache(cacheKey, completion);
      } catch (error) {
        console.error(`Failed to preload cache for prompt: ${prompt}`, error);
      }
    }
  }
}

// Singleton instance
export const performanceOptimizer = new PerformanceOptimizer();

// Initialize preloading
performanceOptimizer.preloadCache();

export default performanceOptimizer;