import { spawn, spawnSync, ChildProcessWithoutNullStreams } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export class ArcticEmbeddingService {
  private modelPath: string;
  private binaryPath: string;
  private persistentProcess: ChildProcessWithoutNullStreams | null = null;
  private processQueue: Array<{
    resolve: (value: number[] | PromiseLike<number[]>) => void,
    reject: (reason?: any) => void,
    text: string
  }> = [];
  private processing: boolean = false;

  constructor(modelPath: string = './arctic_model.onnx', binaryPath: string = './arctic_embed_test') {
    this.modelPath = modelPath;
    this.binaryPath = binaryPath;

    // Validate that the model file exists
    if (!fs.existsSync(this.modelPath)) {
      throw new Error(`Model file not found: ${this.modelPath}`);
    }

    // Validate that the binary file exists
    if (!fs.existsSync(this.binaryPath)) {
      throw new Error(`Binary file not found: ${this.binaryPath}`);
    }
  }

  /**
   * Sanitize text input to prevent command injection
   * @param text Input text to sanitize
   * @returns Sanitized text
   */
  private sanitizeText(text: string): string {
    // Remove potentially dangerous characters
    return text
      .replace(/"/g, '\\"')           // Escape double quotes
      .replace(/'/g, "\\'")           // Escape single quotes
      .replace(/\$/g, '\\$')          // Escape dollar signs
      .replace(/`/g, '\\`')           // Escape backticks
      .replace(/;/g, '')              // Remove semicolons
      .replace(/\|\|/g, '')           // Remove OR operators
      .replace(/&&/g, '')             // Remove AND operators
      .replace(/\n/g, ' ')            // Replace newlines with spaces
      .replace(/\r/g, ' ')            // Replace carriage returns with spaces
      .trim();
  }

  /**
   * Start a persistent process for embedding generation
   */
  private startPersistentProcess(): void {
    if (this.persistentProcess) {
      return; // Process already running
    }

    this.persistentProcess = spawn(this.binaryPath, [this.modelPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: false,
      env: process.env
    });

    let buffer = '';

    this.persistentProcess.stdout.on('data', (data) => {
      buffer += data.toString();

      // Look for complete JSON arrays in the buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const embedding = JSON.parse(line.trim()) as number[];
            if (Array.isArray(embedding) && embedding.length > 0) {
              const nextRequest = this.processQueue.shift();
              if (nextRequest) {
                nextRequest.resolve(embedding);
              }
            }
          } catch (parseError) {
            const nextRequest = this.processQueue.shift();
            if (nextRequest) {
              nextRequest.reject(new Error(`Failed to parse embedding from Arctic binary output: ${line}`));
            }
          }
        }
      }
    });

    this.persistentProcess.stderr.on('data', (data) => {
      console.error(`Arctic binary stderr: ${data}`);
    });

    this.persistentProcess.on('close', (code) => {
      console.warn(`Arctic binary process closed with code ${code}`);
      this.persistentProcess = null;

      // Reject all pending requests
      while (this.processQueue.length > 0) {
        const nextRequest = this.processQueue.shift();
        if (nextRequest) {
          nextRequest.reject(new Error(`Arctic binary process closed unexpectedly with code ${code}`));
        }
      }
    });

    this.persistentProcess.on('error', (error) => {
      console.error(`Arctic binary process error: ${error.message}`);
      this.persistentProcess = null;

      // Reject all pending requests
      while (this.processQueue.length > 0) {
        const nextRequest = this.processQueue.shift();
        if (nextRequest) {
          nextRequest.reject(error);
        }
      }
    });
  }

  /**
   * Stop the persistent process
   */
  private stopPersistentProcess(): void {
    if (this.persistentProcess) {
      this.persistentProcess.kill();
      this.persistentProcess = null;
    }
  }

  /**
   * Generate embeddings using local Arctic C++ binary
   * @param text Input text to embed
   * @returns Promise resolving to embedding vector
   */
  async getEmbedding(text: string): Promise<number[]> {
    // Sanitize the input text
    const sanitizedText = this.sanitizeText(text);

    return new Promise<number[]>((resolve, reject) => {
      // Add request to queue
      this.processQueue.push({ resolve, reject, text: sanitizedText });

      // Start persistent process if not already running
      if (!this.persistentProcess) {
        this.startPersistentProcess();
      }

      // Process the queue if not already processing
      if (!this.processing) {
        this.processNext();
      }
    });
  }

  /**
   * Process the next item in the queue
   */
  private async processNext(): Promise<void> {
    if (this.processQueue.length === 0 || !this.persistentProcess) {
      this.processing = false;
      return;
    }

    this.processing = true;
    const request = this.processQueue[0]; // Peek at the first item

    try {
      // Send the text to the persistent process
      if (this.persistentProcess.stdin.writable) {
        this.persistentProcess.stdin.write(request.text + '\n');
      } else {
        request.reject(new Error('Cannot write to Arctic binary process'));
        this.processQueue.shift(); // Remove the processed item
      }
    } catch (error) {
      request.reject(error);
      this.processQueue.shift(); // Remove the processed item
    }

    // Wait for the response to be handled by the stdout listener
    // Then process the next item
    setTimeout(() => {
      this.processing = false;
      this.processNext(); // Process next item in queue
    }, 10); // Small delay to allow for processing
  }

  /**
   * Generate embeddings for multiple texts using batch processing
   * @param texts Array of input texts to embed
   * @returns Promise resolving to array of embedding vectors
   */
  async getEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    // For batch processing, we'll use Promise.all to process all embeddings concurrently
    // but with rate limiting to avoid overwhelming the system
    const batchSize = 5; // Limit concurrent requests
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => this.getEmbedding(text));
      const batchResults = await Promise.all(batchPromises.map(p => p.catch(e => ({ error: e }))));

      // Check for errors and handle appropriately
      for (const result of batchResults) {
        if ('error' in result) {
          throw result.error;
        } else {
          results.push(result);
        }
      }
    }

    return results;
  }

  /**
   * Clean up resources
   */
  async cleanup(): Promise<void> {
    this.stopPersistentProcess();
  }
}

// Example usage for the memory-lancedb plugin
export class MemoryLanceDBPlugin {
  private embeddingService: ArcticEmbeddingService;

  constructor(modelPath?: string, binaryPath?: string) {
    this.embeddingService = new ArcticEmbeddingService(modelPath, binaryPath);
  }

  async storeMemory(text: string, metadata?: Record<string, any>) {
    try {
      // Generate embedding using local Arctic binary instead of OpenAI API
      const embedding = await this.embeddingService.getEmbedding(text);

      // Store in LanceDB with the generated embedding
      // Implementation would connect to LanceDB and store the vector
      console.log(`Storing memory with embedding of length ${embedding.length}`);

      // Placeholder for actual LanceDB storage logic
      // const db = await lancedb.connect(this.dbPath);
      // const table = await db.openTable('memories');
      // await table.add([{ vector: embedding, text, metadata, timestamp: Date.now() }]);

      return { success: true, embeddingLength: embedding.length };
    } catch (error) {
      console.error('Error storing memory:', error);
      throw error;
    }
  }

  async searchMemories(query: string, limit: number = 5) {
    try {
      // Generate embedding for the query using local Arctic binary
      const queryEmbedding = await this.embeddingService.getEmbedding(query);

      // Search in LanceDB using the generated embedding
      // Implementation would connect to LanceDB and perform vector search
      console.log(`Searching memories with query embedding of length ${queryEmbedding.length}`);

      // Placeholder for actual LanceDB search logic
      // const db = await lancedb.connect(this.dbPath);
      // const table = await db.openTable('memories');
      // const results = await table.search(queryEmbedding).limit(limit).execute();

      return { results: [], queryEmbeddingLength: queryEmbedding.length };
    } catch (error) {
      console.error('Error searching memories:', error);
      throw error;
    }
  }

  /**
   * Clean up resources when done
   */
  async cleanup(): Promise<void> {
    await this.embeddingService.cleanup();
  }
}

// Export for use in the OpenClaw ecosystem
export default MemoryLanceDBPlugin;