# Azure AI Search Document Uploader for RAG

## Overview

This utility processes documents from a directory, intelligently chunks them, and uploads them to an Azure AI Search index for use with Retrieval-Augmented Generation (RAG) chatbots. The script handles various document formats, extracting text and metadata while preserving document structure.

## Features

- Processes multiple document formats (PDF, DOCX, TXT, HTML, CSV, PPTX, XLSX, etc.)
- Intelligent chunking based on document structure (headings, sections, pages)
- Vector embeddings generation using Azure OpenAI
- Support for both vector search and semantic search
- Document metadata extraction and preservation
- Configurable chunk sizes and overlap
- Language detection
- Advanced pattern-based section detection for better chunking
- Support for both text-embedding-ada-002 and text-embedding-3-large models
- Automatic vector dimension adjustment based on the embedding model
- Regex-based file filtering for inclusion/exclusion

## Prerequisites

- Python 3.6+
- Azure AI Search instance
- Azure OpenAI instance (if using vector search)

## Required Python Libraries

```bash
pip install azure-search-documents azure-core PyPDF2 python-docx pandas bs4 python-pptx langdetect requests pymupdf
```

## Usage

```bash
python rag_upload.py --folder <documents-folder> --endpoint <search-endpoint> --key <search-api-key> --index <index-name> [options]
```

### Required Arguments

- `--folder`: Path to the folder containing documents to process
- `--endpoint`: Azure AI Search service endpoint URL
- `--key`: Azure AI Search admin API key
- `--index`: Name of the search index to create or update

### Optional Arguments

- `--chunk-size`: Size of text chunks in characters (default: 1000)
- `--chunk-overlap`: Overlap between chunks in characters (default: 100)
- `--vector-search`: Enable vector search capabilities
- `--semantic-search`: Enable semantic search capabilities
- `--aoai-endpoint`: Azure OpenAI endpoint for generating embeddings
- `--aoai-key`: Azure OpenAI API key
- `--aoai-deployment`: Azure OpenAI deployment name for embeddings model
- `--vector-dimensions`: Dimensions for vector embeddings (default: 1536 for ada-002, 3072 for text-embedding-3-large)
- `--skip-vectors`: Skip vector processing even if vector search is enabled
- `--force-recreate`: Force recreation of the search index
- `--include`: File patterns to include (regex, can specify multiple)
- `--exclude`: File patterns to exclude (regex, can specify multiple)

## Example Commands

### Basic Usage

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex
```

### With Vector Search (text-embedding-3-large)

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

### With Vector Search (text-embedding-ada-002)

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-ada-002
```

### With Semantic Search and Custom Chunking

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --semantic-search --chunk-size 1500 --chunk-overlap 150
```

### Process Only Specific File Types

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --include "\.pdf$" "\.docx$"
```

## Common Use Cases and Examples

### Processing Technical Documentation

When processing technical documentation with code snippets and structured content:

```bash
python rag_upload.py --folder "C:\Documents\technical-docs" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index techdocs --chunk-size 1200 --chunk-overlap 200 --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large --include "\.md$" "\.pdf$" "\.py$" "\.ipynb$"
```

### Processing Legal Documents

For legal documents where preserving context is critical:

```bash
python rag_upload.py --folder "C:\Documents\legal" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index legal-docs --chunk-size 2000 --chunk-overlap 300 --semantic-search --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

### Processing Product Manuals

For product manuals and guides with images and tables:

```bash
python rag_upload.py --folder "C:\Documents\product-manuals" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index product-docs --chunk-size 1500 --chunk-overlap 150 --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large --exclude "\.jpg$" "\.png$" "\.gif$"
```

### Processing Research Papers

For scientific and research papers with complex content:

```bash
python rag_upload.py --folder "C:\Documents\research" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index research-papers --chunk-size 1800 --chunk-overlap 250 --vector-search --semantic-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

### Processing Multiple Document Types with Different Chunking Strategies

When you need to process different document types with different chunking strategies, you can run the script multiple times with different parameters:

```bash
# First process text-heavy documents with smaller chunks
python rag_upload.py --folder "C:\Documents\mixed" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index mixed-content --chunk-size 800 --chunk-overlap 100 --include "\.txt$" "\.md$" --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large

# Then process structured documents with larger chunks
python rag_upload.py --folder "C:\Documents\mixed" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index mixed-content --chunk-size 1500 --chunk-overlap 200 --include "\.pdf$" "\.docx$" --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

### Processing Large Document Collections

For large document collections, process in batches by subfolder:

```bash
# Process first batch
python rag_upload.py --folder "C:\Documents\large-collection\batch1" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index large-collection --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large

# Process second batch
python rag_upload.py --folder "C:\Documents\large-collection\batch2" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index large-collection --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

## Key Components

### Document Processing

The script supports multiple document formats:
- PDF: Extracts text while preserving page information
- DOCX: Processes text, headings, tables
- HTML: Extracts content while preserving semantic structure
- CSV/Excel: Processes tabular data
- PPTX: Extracts slide content and titles
- Text files (TXT, MD, code files): Processes raw text
- JSON: Intelligently chunks based on object structure

### Intelligent Chunking

The script uses different chunking strategies based on document type:
- For PDFs: Attempts to chunk by sections or page boundaries
- For code: Respects function and class boundaries
- For markdown: Splits at heading markers
- For regular text: Uses paragraph and sentence boundaries
- For JSON: Chunks based on object structure
- Advanced pattern-based section detection for better content organization

### Search Index Structure

The created index includes the following fields:
- `id`: Unique document ID
- `content`: The text content chunk
- `file_name`: Original file name
- `file_path`: Path to the original file
- `file_extension`: File extension (.pdf, .docx, etc.)
- `file_size`: Size in bytes
- `last_modified`: Last modification date
- `chunk_id`: Sequential ID of chunk within its document
- `chunk_total`: Total number of chunks in the document
- `language`: Detected language
- `title`: Auto-generated title for the chunk
- `author`: Author information if available
- `content_vector`: Vector embeddings (if vector search is enabled)

## Best Practices

1. **Choose appropriate chunk sizes** for your content type and retrieval needs:
   - Larger chunks (1500-2000 chars) preserve more context but may retrieve irrelevant information
   - Smaller chunks (500-1000 chars) are more precise but may lose context

2. **Vector search configuration**:
   - Use `text-embedding-3-large` deployment for best results (3072 dimensions)
   - Use `text-embedding-ada-002` for compatibility with existing systems (1536 dimensions)
   - The script automatically adjusts vector dimensions based on the model used
   - Ensure your Azure OpenAI service has sufficient rate limits

3. **Processing large document sets**:
   - Process files in batches to avoid timeout issues
   - Use the `--include` parameter to target specific document types

4. **Index optimization**:
   - Use `--force-recreate` when changing index structure
   - Enable `--semantic-search` for better natural language queries

## Troubleshooting

- **Rate limiting errors**: If you encounter rate limit errors with Azure OpenAI, increase the delay between embedding requests or process in smaller batches
- **Index creation failures**: Make sure you have admin permissions for the search service
- **Chunking issues**: Adjust chunk size and overlap for different document types
- **Vector search errors**: Ensure your search service supports vector search capabilities
- **Document format issues**: Try excluding problematic files using the `--exclude` parameter

## Advanced Configuration

For more advanced scenarios, you may need to modify the script directly:

- Custom embedding models: Update the `EmbeddingGenerator` class
- Additional file formats: Add new extractors to the `DocumentProcessor` class
- Custom chunking strategies: Modify the `TextChunker` class

![Flow Diagram](flow_diagram.svg)

## License

This code is provided as-is with no warranties. Use at your own risk.
