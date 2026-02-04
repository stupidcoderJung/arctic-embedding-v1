import { spawn } from 'child_process';
import * as fs from 'fs';

/**
 * ArcticEmbeddings class for generating embeddings using the optimized C++ binary
 */
export class ArcticEmbeddings {
  private modelPath: string;
  private binaryPath: string;

  constructor(
    modelPath: string = './arctic_model.onnx',
    binaryPath: string = './bin/arctic_embed_test'
  ) {
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
   * Generate embeddings using local Arctic C++ binary
   * The binary expects command line arguments: <model_path> <input_text>
   */
  async embedQuery(text: string): Promise<number[]> {
    // Sanitize the input text
    const sanitizedText = this.sanitizeText(text);

    return new Promise<number[]>((resolve, reject) => {
      // Execute the binary with command line arguments
      const child = spawn(this.binaryPath, [this.modelPath, sanitizedText], {
        stdio: ['pipe', 'pipe', 'pipe'],
        shell: false,
        env: process.env
      });

      let stdoutData = '';
      let stderrData = '';

      child.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderrData += data.toString();
      });

      child.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Arctic binary exited with code ${code}: ${stderrData}`));
          return;
        }

        // Parse the JSON output from stdout
        const lines = stdoutData.split('\n');
        for (const line of lines) {
          if (line.trim() && line.startsWith('[') && line.endsWith(']')) {
            try {
              const embedding = JSON.parse(line.trim()) as number[];
              if (Array.isArray(embedding) && embedding.length > 0) {
                // Ensure consistent embedding size by truncating or padding
                const targetSize = 384; // Standard size for Arctic-Embed-Tiny
                let normalizedEmbedding: number[];

                if (embedding.length > targetSize) {
                  // Truncate to target size
                  normalizedEmbedding = embedding.slice(0, targetSize);
                } else if (embedding.length < targetSize) {
                  // Pad with zeros to target size
                  normalizedEmbedding = [...embedding, ...new Array(targetSize - embedding.length).fill(0)];
                } else {
                  // Already the right size
                  normalizedEmbedding = embedding;
                }

                resolve(normalizedEmbedding);
                return;
              }
            } catch (parseError) {
              // Continue to next line if this one fails to parse
              continue;
            }
          }
        }

        reject(new Error(`No valid embedding found in output: ${stdoutData}`));
      });

      child.on('error', (error) => {
        reject(new Error(`Arctic binary execution error: ${error.message}`));
      });
    });
  }

  /**
   * Generate embeddings for multiple texts using batch processing
   * @param texts Array of input texts to embed
   * @returns Promise resolving to array of embedding vectors
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    // Process embeddings sequentially to avoid overwhelming the system
    const results: number[][] = [];
    for (const text of texts) {
      const embedding = await this.embedQuery(text);
      results.push(embedding);
    }

    return results;
  }

  /**
   * Clean up resources
   */
  async cleanup(): Promise<void> {
    // Nothing to clean up since we're not using a persistent process
  }
}