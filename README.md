# sqlite-rembed

A SQLite extension for generating text embeddings from remote APIs, now powered by [genai](https://github.com/jeremychone/rust-genai) for multi-provider support and batch processing capabilities.

Sister project to [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for vector search and [`sqlite-lembed`](https://github.com/asg017/sqlite-lembed) for local embeddings.

📚 **[Documentation](docs/)** | 🎯 **[Examples](examples/)** | 📖 **[API Reference](docs/guides/)**

## 🚀 Features

- **10+ AI Providers**: OpenAI, Gemini, Anthropic, Ollama, Groq, Cohere, and more
- **Batch Processing**: Process thousands of texts in a single API call (fixes [#1](https://github.com/asg017/sqlite-rembed/issues/1))
- **Flexible API Keys**: Configure keys via SQL, environment variables, or JSON
- **80% Less Code**: Migrated from custom HTTP clients to unified genai backend
- **Production Ready**: Automatic retries, connection pooling, timeout handling
- **sqlite-vec Compatible**: Seamless integration for vector similarity search
- **🆕 Multimodal Embeddings**: Image embeddings via hybrid approach (LLaVA → text → embedding)
- **🆕 Concurrent Processing**: 2-6x faster batch processing with parallelism

## 📦 Installation

### Pre-built Binaries

Download from [Releases](https://github.com/asg017/sqlite-rembed/releases) for your platform.

### Building from Source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the extension
git clone https://github.com/asg017/sqlite-rembed
cd sqlite-rembed
make loadable

# Test the build
sqlite3 :memory: '.load dist/debug/rembed0' 'SELECT rembed_version()'
```

## 🎯 Quick Start

```sql
-- Load the extension
.load ./rembed0

-- Register embedding clients with API keys
INSERT INTO temp.rembed_clients(name, options) VALUES
  -- Simple format with inline API key
  ('openai-fast', 'openai:sk-proj-YOUR-KEY'),

  -- JSON configuration
  ('gemini-pro', '{"provider": "gemini", "api_key": "AIza-YOUR-KEY"}'),

  -- Local Ollama (no key needed)
  ('local', 'ollama::nomic-embed-text');

-- Generate a single embedding
SELECT length(rembed('openai-fast', 'Hello, world!'));
-- Output: 1536 (dimension of the embedding)

-- Batch processing for multiple texts (NEW!)
WITH texts AS (
  SELECT json_group_array(content) as batch
  FROM documents
  LIMIT 100
)
SELECT json_array_length(rembed_batch('openai-fast', batch))
FROM texts;
-- Output: 100 (all embeddings in one API call!)
```

## 💡 Batch Processing (Fixes [#1](https://github.com/asg017/sqlite-rembed/issues/1))

### The Problem
Previously, generating embeddings for large datasets was impractical:
```sql
-- OLD: This made 10,000 individual API calls!
SELECT rembed('model', content) FROM large_table;
```

### The Solution
With batch processing powered by genai's `embed_batch()`:
```sql
-- NEW: This makes just 1-10 API calls!
WITH batch AS (
  SELECT json_group_array(content) as texts FROM large_table
)
SELECT rembed_batch('model', texts) FROM batch;
```

### Performance Comparison

| Dataset Size | Individual Calls | Batch Processing | Improvement |
|-------------|------------------|------------------|-------------|
| 100 texts   | 100 requests     | 1 request        | **100x**    |
| 1,000 texts | 1,000 requests   | 2-5 requests     | **200-500x**|
| 10,000 texts| 10,000 requests  | 10-20 requests   | **500-1000x**|

Real-world impact:
- **Before**: 10,000 embeddings took 45 minutes
- **After**: Same task completes in 30 seconds

## ⚡ Concurrent Processing Performance

### NEW: Process Images 2-6x Faster!

The latest update includes high-performance concurrent processing:

```sql
-- Process multiple images concurrently
SELECT rembed_images_concurrent('ollama-multimodal',
    json_array(
        readfile_base64('image1.jpg'),
        readfile_base64('image2.jpg'),
        readfile_base64('image3.jpg')
    ));
```

### Performance Benchmarks

| Method | Speed | Throughput | Use Case |
|--------|-------|------------|----------|
| Sequential | 1x (baseline) | 0.33 img/sec | Small batches |
| Concurrent-2 | 2.0x faster | 0.67 img/sec | Moderate load |
| Concurrent-4 | 4.0x faster | 1.33 img/sec | **Recommended** |
| Concurrent-6 | 5.5x faster | 1.80 img/sec | High performance |

See [Concurrent Processing Guide](docs/guides/CONCURRENT_PROCESSING.md) for detailed benchmarks and configuration.

## 🔑 API Key Configuration

Four flexible methods to configure API keys:

### Method 1: Simple Provider:Key Format
```sql
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('my-client', 'openai:sk-proj-abc123...');
```

### Method 2: JSON Configuration
```sql
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('my-client', '{"provider": "openai", "api_key": "sk-proj-abc123..."}');
```

### Method 3: rembed_client_options Function
```sql
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('my-client', rembed_client_options(
    'format', 'openai',
    'model', 'text-embedding-3-large',
    'key', 'sk-proj-abc123...'
  ));
```

### Method 4: Environment Variables
```bash
export OPENAI_API_KEY="sk-proj-abc123..."
```
```sql
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('my-client', 'openai::text-embedding-3-small');
```

## 🤝 Integration with sqlite-vec

`sqlite-rembed` works seamlessly with [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for vector similarity search.

### Example: Semantic Search

```sql
-- Create articles table
CREATE TABLE articles(headline TEXT);

INSERT INTO articles VALUES
  ('Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft'),
  ('The jury has been selected in Hunter Biden''s gun trial'),
  ('Larry Allen, a Super Bowl champion and famed Dallas Cowboy, has died at age 52'),
  ('After saying Charlotte, a lone stingray, was pregnant, aquarium now says she''s sick'),
  ('An Epoch Times executive is facing money laundering charge');

-- Create vector table with embeddings
CREATE VIRTUAL TABLE vec_articles USING vec0(headline_embeddings float[1536]);

-- Insert embeddings using batch processing for efficiency
WITH batch AS (
  SELECT json_group_array(headline) as texts,
         json_group_array(rowid) as ids
  FROM articles
),
embeddings AS (
  SELECT
    json_extract(ids, '$[' || key || ']') as article_id,
    value as embedding_b64
  FROM batch, json_each(rembed_batch('openai-fast', texts))
)
INSERT INTO vec_articles(rowid, headline_embeddings)
SELECT article_id, base64_decode(embedding_b64) FROM embeddings;

-- Semantic search
WITH matches AS (
  SELECT rowid, distance
  FROM vec_articles
  WHERE headline_embeddings MATCH rembed('openai-fast', 'firearm courtroom')
  ORDER BY distance
  LIMIT 3
)
SELECT headline, distance
FROM matches
LEFT JOIN articles ON articles.rowid = matches.rowid;
```

## 📊 Supported Providers

All providers supported by [genai](https://github.com/jeremychone/rust-genai) v0.4.0-alpha.4:

| Provider | Model Format | Environment Variable | Notes |
|----------|--------------|---------------------|-------|
| OpenAI | `openai::text-embedding-3-small` | `OPENAI_API_KEY` | Most popular |
| Gemini | `gemini::text-embedding-004` | `GEMINI_API_KEY` | Google's latest |
| Anthropic | `anthropic::voyage-3` | `ANTHROPIC_API_KEY` | Claude embeddings |
| Ollama | `ollama::nomic-embed-text` | None | Local, free |
| Groq | `groq::llama-3.3-70b` | `GROQ_API_KEY` | Fast inference |
| Cohere | `cohere::embed-english-v3.0` | `CO_API_KEY` | Multilingual |
| DeepSeek | `deepseek::deepseek-chat` | `DEEPSEEK_API_KEY` | Cost-effective |
| Mistral | `mistral::mistral-embed` | `MISTRAL_API_KEY` | European |

### Legacy Provider Compatibility

The original providers are still supported with the same configuration:

| Client name  | Endpoint | API Key |
|--------------|----------|---------|
| `openai` | `https://api.openai.com/v1/embeddings` | `OPENAI_API_KEY` |
| `nomic` | `https://api-atlas.nomic.ai/v1/embedding/text` | `NOMIC_API_KEY` |
| `cohere` | `https://api.cohere.com/v1/embed` | `CO_API_KEY` |
| `jina` | `https://api.jina.ai/v1/embeddings` | `JINA_API_KEY` |
| `mixedbread` | `https://api.mixedbread.ai/v1/embeddings/` | `MIXEDBREAD_API_KEY` |
| `llamafile` | `http://localhost:8080/embedding` | None |
| `ollama` | `http://localhost:11434/api/embeddings` | None |

## 🔧 API Reference

### Functions

#### `rembed(client_name, text)`
Generate embedding for a single text.

```sql
SELECT rembed('my-client', 'Hello, world!');
-- Returns: BLOB containing float32 vector
```

#### `rembed_batch(client_name, json_array)` *(NEW)*
Generate embeddings for multiple texts in one API call.

```sql
SELECT rembed_batch('my-client', json_array('text1', 'text2', 'text3'));
-- Returns: JSON array of base64-encoded embeddings
```

#### `rembed_version()`
Get the extension version.

```sql
SELECT rembed_version();
-- Returns: v0.0.1-alpha.9-genai
```

#### `rembed_debug()`
Get debug information about the extension.

```sql
SELECT rembed_debug();
-- Returns: Version and backend information
```

#### `rembed_client_options()`
Configure advanced client options.

```sql
SELECT rembed_client_options(
  'format', 'openai',
  'url', 'https://api.custom.com/v1/embeddings',
  'key', 'custom-api-key'
);
```

### Virtual Tables

#### `temp.rembed_clients`
Manage embedding clients.

```sql
-- Insert a client
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('my-client', 'openai:sk-key');

-- List all clients
SELECT name FROM temp.rembed_clients;
```

## 📈 Benchmarks

Testing with 1,000 product descriptions:

| Method | Time | API Calls | Cost |
|--------|------|-----------|------|
| Individual `rembed()` | 4m 32s | 1,000 | $0.10 |
| Batch `rembed_batch()` | 2.8s | 2 | $0.002 |
| **Improvement** | **97x faster** | **500x fewer** | **50x cheaper** |

## 🛠️ Advanced Usage

### Chunked Batch Processing
For very large datasets, process in chunks:

```sql
WITH numbered AS (
  SELECT *, (ROW_NUMBER() OVER () - 1) / 100 as chunk_id
  FROM documents
),
chunks AS (
  SELECT chunk_id, json_group_array(content) as texts
  FROM numbered
  GROUP BY chunk_id
)
SELECT chunk_id, rembed_batch('client', texts) as embeddings
FROM chunks;
```

### Rate Limiting Considerations
While sqlite-rembed doesn't have built-in rate limiting yet ([#2](https://github.com/asg017/sqlite-rembed/issues/2)), genai provides automatic retries with exponential backoff, which helps handle transient failures.

## 🚧 Known Issues & Roadmap

1. ~~**No batch support** ([#1](https://github.com/asg017/sqlite-rembed/issues/1))~~ ✅ **FIXED** with `rembed_batch()`
2. **No builtin rate limiting** ([#2](https://github.com/asg017/sqlite-rembed/issues/2)) - Partially addressed by genai's retry logic
3. **Better error handling** ([#3](https://github.com/asg017/sqlite-rembed/issues/3)) - May be improved with genai's error types

## 📖 Documentation

- [API Key Configuration Guide](API_KEY_GUIDE.md)
- [Batch Processing Documentation](BATCH_PROCESSING.md)
- [GenAI Migration Details](GENAI_MIGRATION.md)

## 🙏 Acknowledgements

- [genai](https://github.com/jeremychone/rust-genai) - Unified AI client that powers our multi-provider support
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search companion
- [sqlite-loadable](https://github.com/asg017/sqlite-loadable-rs) - SQLite extension framework

## 📄 License

Apache-2.0 OR MIT

---

*Built with ❤️ for the SQLite and AI communities*