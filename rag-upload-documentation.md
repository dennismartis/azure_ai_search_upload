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

## Prerequisites

- Python 3.6+
- Azure AI Search instance
- Azure OpenAI instance (if using vector search)

## Required Python Libraries

```bash
pip install azure-search-documents PyPDF2 python-docx pandas bs4 python-pptx langdetect requests
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
- `--vector-dimensions`: Dimensions for vector embeddings (default: 1536)
- `--skip-vectors`: Skip vector processing even if vector search is enabled
- `--force-recreate`: Force recreation of the search index
- `--include`: File patterns to include (regex, can specify multiple)
- `--exclude`: File patterns to exclude (regex, can specify multiple)

## Example Commands

### Basic Usage

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex
```

### With Vector Search

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --vector-search --aoai-endpoint https://myopenai.openai.azure.com/ --aoai-key YOUR_OPENAI_KEY --aoai-deployment text-embedding-3-large
```

### With Semantic Search and Custom Chunking

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --semantic-search --chunk-size 1500 --chunk-overlap 150
```

### Process Only Specific File Types

```bash
python rag_upload.py --folder "C:\Documents\rag" --endpoint https://mysearch.search.windows.net --key YOUR_SEARCH_KEY --index myindex --include "\.pdf$" "\.docx$"
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

### Intelligent Chunking

The script uses different chunking strategies based on document type:
- For PDFs: Attempts to chunk by sections or page boundaries
- For code: Respects function and class boundaries
- For markdown: Splits at heading markers
- For regular text: Uses paragraph and sentence boundaries

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
   - Use `text-embedding-3-large` deployment for best results
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

## License

This code is provided as-is with no warranties. Use at your own risk.
