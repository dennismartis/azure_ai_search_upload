#!/usr/bin/env python3
"""
Azure AI Search Document Uploader for RAG

This script processes documents from a directory, chunks them intelligently,
and uploads them to an Azure AI Search index for use with RAG chatbots.

run with:

"""
# python .\rag_upload.py --folder "C:\Users\marti\Downloads\Documents\rag"  --endpoint https://dennis11search.search.windows.net  --key <SEARCH_API_KEY>  --index myindex --vector-search --aoai-endpoint https://dennisopenai.openai.azure.com/ --aoai-key <OPENAI_API_KEY>  --aoai-deployment text-embedding-3-large --force-recreate
import os
import argparse
import json
import logging
import hashlib
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

# Azure libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchFieldDataType,
    SimpleField, SearchableField
)

# Document processing libraries - install with pip
import requests
import PyPDF2
import docx
import pandas as pd
from bs4 import BeautifulSoup
import pptx
from langdetect import detect

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AzureSearchConfig:
    """Configuration for Azure AI Search service"""
    endpoint: str
    api_key: str
    index_name: str
    vector_search: bool = False
    semantic_search: bool = False
    vector_dimensions: int = 1536
    force_recreate: bool = False

@dataclass
class OpenAIConfig:
    """Configuration for Azure OpenAI service"""
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    skip_embeddings: bool = False

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    folder_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 100
    include_filters: List[str] = field(default_factory=list)
    exclude_filters: List[str] = field(default_factory=list)

class DocumentProcessor:
    """Base class for document processors"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file and extract its text content and metadata"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        # Format the last_modified date in ISO 8601 format with timezone info
        try:
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            last_modified_iso = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format with 'Z' for UTC
        except:
            # Use current time if file modification time cannot be retrieved
            last_modified_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Basic metadata for all files
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": file_size,
            "file_extension": file_extension,
            "last_modified": last_modified_iso
        }
        
        # Get the appropriate processor for this file type
        processor = self._get_processor_for_file(file_path)
        
        # Extract text and metadata from file
        try:
            result = processor(file_path)
            
            # Ensure all date fields in metadata have valid format
            for key, value in result["metadata"].items():
                if isinstance(value, str) and 'date' in key.lower() and value == '':
                    # Use the modification date as a fallback for any empty date fields
                    result["metadata"][key] = last_modified_iso
            
            # Combine file metadata with extracted metadata
            result["metadata"].update(metadata)
            
            return result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"text": "", "metadata": metadata}
    
    def _get_processor_for_file(self, file_path: str) -> callable:
        """Get the appropriate processor function for a file type"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        processors = {
            '.pdf': self._extract_text_from_pdf,
            '.docx': self._extract_text_from_docx,
            '.txt': self._extract_text_from_txt,
            '.md': self._extract_text_from_txt,
            '.py': self._extract_text_from_txt,
            '.js': self._extract_text_from_txt,
            '.html': self._extract_text_from_html,
            '.htm': self._extract_text_from_html,
            '.csv': self._extract_text_from_csv,
            '.pptx': self._extract_text_from_pptx,
            '.xlsx': self._extract_text_from_xlsx,
            '.xls': self._extract_text_from_xlsx,
        }
        
        return processors.get(file_extension, lambda f: {"text": "", "metadata": {}})
    
    def _extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF files with semantic awareness
        """
        text = ""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        if value and isinstance(value, str):
                            metadata[key] = value
                
                # Extract all text as a continuous document, but preserve some page markers
                full_text = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Add a subtle page marker that won't disrupt semantic chunking
                        processed_text = f"{page_text.strip()}\n[pg:{page_num + 1}]\n\n"
                        full_text.append(processed_text)
                        
                # Store page count in metadata
                metadata["pdf_page_count"] = len(pdf_reader.pages)
                
                # Join all text into a single document
                text = "".join(full_text)
                
                # Extract potential section headers for better chunking
                headers = re.findall(r'\n([A-Z][A-Z\s]{3,}[A-Z])[^a-z]', text)
                if headers:
                    metadata["potential_sections"] = headers
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from DOCX files
        """
        text = ""
        metadata = {}
        
        try:
            doc = docx.Document(file_path)
            
            # Extract document properties if available
            core_properties = getattr(doc, 'core_properties', None)
            if core_properties:
                metadata["author"] = getattr(core_properties, "author", "")
                metadata["created"] = getattr(core_properties, "created", "")
                metadata["last_modified_by"] = getattr(core_properties, "last_modified_by", "")
                metadata["title"] = getattr(core_properties, "title", "")
                metadata["subject"] = getattr(core_properties, "subject", "")
            
            # Extract headings and content
            for para in doc.paragraphs:
                if para.style and para.style.name.startswith('Heading'):
                    # Add extra newlines for headings to improve chunking
                    text += f"\n{para.text}\n\n"
                else:
                    text += para.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text += row_text + "\n"
                text += "\n"
                
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_txt(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from TXT files
        """
        text = ""
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        last_modified_iso = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format with 'Z' for UTC
        
        metadata = {
            "last_modified": last_modified_iso
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from CSV files
        """
        text = ""
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        last_modified_iso = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format with 'Z' for UTC
        
        metadata = {
            "last_modified": last_modified_iso
        }
        
        try:
            df = pd.read_csv(file_path)
            # Get headers
            headers = df.columns.tolist()
            metadata["headers"] = ", ".join(headers)
            
            # Convert to readable text
            text = df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error extracting text from CSV {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_html(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from HTML files
        """
        text = ""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata from meta tags
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    metadata[meta['name']] = meta['content']
            
            # Extract title
            title = soup.find('title')
            if title:
                metadata['title'] = title.text
                text += f"Title: {title.text}\n\n"
            
            # Extract headings and content with structure
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text += f"\n{heading.text.strip()}\n\n"
                
                # Get content until the next heading
                for sibling in heading.next_siblings:
                    if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    if sibling.name and sibling.name not in ['script', 'style']:
                        content = sibling.get_text(strip=True)
                        if content:
                            text += content + "\n"
            
            # Fallback: if we didn't get much text, extract all text
            if len(text) < 200:
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator="\n")
        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_pptx(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PowerPoint PPTX files
        """
        text = ""
        metadata = {}
        
        try:
            prs = pptx.Presentation(file_path)
            
            # Extract metadata if available
            if hasattr(prs.core_properties, 'author'):
                metadata["author"] = prs.core_properties.author
            if hasattr(prs.core_properties, 'title'):
                metadata["title"] = prs.core_properties.title
            
            # Extract text from slides
            for i, slide in enumerate(prs.slides):
                text += f"\n--- Slide {i+1} ---\n"
                
                # Extract slide title if available
                if slide.shapes.title:
                    text += f"Title: {slide.shapes.title.text}\n\n"
                
                # Extract text from all shapes in the slide
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text += shape.text + "\n"
                
                text += "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PPTX {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def _extract_text_from_xlsx(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from Excel XLSX files
        """
        text = ""
        metadata = {}
        
        try:
            # Use pandas to read Excel
            xlsx = pd.ExcelFile(file_path)
            metadata["sheet_names"] = ", ".join(xlsx.sheet_names)
            
            # Process each sheet
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name)
                
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += df.to_string(index=False) + "\n\n"
        except Exception as e:
            logger.error(f"Error extracting text from XLSX {file_path}: {e}")
        
        return {
            "text": text,
            "metadata": metadata
        }

class TextChunker:
    """Class for chunking text into manageable pieces"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with specified size and overlap, respecting content structure
        """
        # If the text is short enough, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        
        # Try to detect if this is code or structured data
        is_code = bool(re.search(r'(def |class |function |import |from |var |const |let |public |private |#include)', text))
        is_json = bool(text.strip().startswith('{') and text.strip().endswith('}'))
        is_markdown = bool(re.search(r'^#{1,6} |\n#{1,6} |```|\*\*|__|\[.+\]\(.+\)', text))
        
        # Check if this appears to be a PDF with potential section headers
        has_pdf_markers = '[pg:' in text and re.search(r'\[pg:\d+\]', text)
        
        if has_pdf_markers:
            # Try to find section markers or headers for better chunking of PDFs
            # Common patterns in PDFs include: ALL CAPS HEADINGS, Chapter X, Section X.X, etc.
            sections = []
            
            # Check for various heading patterns
            heading_patterns = [
                r'\n([A-Z][A-Z\s]{3,}[A-Z])[^a-z]',  # ALL CAPS HEADINGS
                r'\n(Chapter\s+\d+[.:]\s*\w+)',       # Chapter headings
                r'\n((?:\d+\.){1,3}\s*\w+)',          # Section numbers like 1.2.3
                r'\n([IVX]+\.\s+\w+)',                # Roman numeral sections
                r'\b(\d+\.\d+\s+[A-Z][a-z]+)'         # Numbered headings like "1.2 Introduction"
            ]
            
            # Try each pattern to find potential section breaks
            for pattern in heading_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Get position of the heading
                    pos = match.start()
                    sections.append(pos)
            
            # If we found sections, use them for chunking
            if sections:
                sections.sort()
                start_pos = 0
                
                # Create chunks based on section boundaries
                for section_pos in sections:
                    # If adding this section would exceed chunk size, break here
                    if section_pos - start_pos > self.chunk_size and start_pos < section_pos:
                        chunk_text = text[start_pos:section_pos].strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                        start_pos = section_pos
                
                # Add the last section
                if start_pos < len(text):
                    chunk_text = text[start_pos:].strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                
                # If we successfully created chunks, return them
                if chunks:
                    return chunks
            
            # If no clear sections were found, fall back to chunking by paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para) + 2  # +2 for the newlines
                
                # If this paragraph would exceed the chunk size and we already have content,
                # complete the current chunk and start a new one
                if current_size + para_size > self.chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap by including some previous paragraphs
                    overlap_size = 0
                    overlap_paras = []
                    for prev_para in reversed(current_chunk):
                        if overlap_size + len(prev_para) + 2 <= self.chunk_overlap:
                            overlap_paras.insert(0, prev_para)
                            overlap_size += len(prev_para) + 2
                        else:
                            break
                    
                    current_chunk = overlap_paras
                    current_size = overlap_size
                
                current_chunk.append(para)
                current_size += para_size
            
            # Add the last chunk if it has content
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # If we successfully created chunks, return them
            if chunks:
                return chunks
        
        if is_json:
            # For JSON, try to split at the top-level objects
            try:
                data = json.loads(text)
                if isinstance(data, list) and len(data) > 1:
                    # If it's a list of objects, split by list items
                    for i in range(0, len(data), max(1, self.chunk_size // 500)):
                        subset = data[i:min(i + max(1, self.chunk_size // 500), len(data))]
                        chunks.append(json.dumps(subset, indent=2))
                    return chunks
                elif isinstance(data, dict):
                    # For large dictionaries, split by top-level keys
                    keys = list(data.keys())
                    for i in range(0, len(keys), max(1, self.chunk_size // 500)):
                        subset_keys = keys[i:min(i + max(1, self.chunk_size // 500), len(keys))]
                        subset = {k: data[k] for k in subset_keys}
                        chunks.append(json.dumps(subset, indent=2))
                    return chunks
            except:
                pass  # If JSON parsing fails, fall back to regular chunking
        
        if is_code:
            # For code, try to split at function boundaries
            lines = text.split('\n')
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line_with_newline = line + '\n'
                line_length = len(line_with_newline)
                
                # If adding this line would exceed chunk size and we already have content
                if current_length + line_length > self.chunk_size and current_chunk:
                    chunks.append(''.join(current_chunk))
                    # Keep some context for the next chunk (overlap)
                    overlap_lines = current_chunk[-min(len(current_chunk), self.chunk_overlap // 20):]
                    current_chunk = overlap_lines
                    current_length = sum(len(l) + 1 for l in overlap_lines)
                
                current_chunk.append(line_with_newline)
                current_length += line_length
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            return chunks
        
        if is_markdown:
            # For markdown, try to split at heading boundaries
            sections = re.split(r'(\n#{1,6} )', text)
            current_chunk = []
            current_length = 0
            
            for i in range(len(sections)):
                section = sections[i]
                section_length = len(section)
                
                if current_length + section_length > self.chunk_size and current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(section)
                current_length += section_length
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            return chunks
        
        # Default chunking for regular text - use paragraphs and sentences
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para) + 2  # +2 for the newlines
            
            # If this paragraph would exceed the chunk size on its own, split it by sentences
            if para_size > self.chunk_size:
                # Process this large paragraph separately
                if current_chunk:
                    # First, complete the current chunk
                    chunks.append("\n\n".join(current_chunk).strip())
                    current_chunk = []
                    current_size = 0
                
                # Split this large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_sentence_chunk = []
                current_sentence_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence) + 1  # +1 for the space
                    
                    if current_sentence_size + sentence_size > self.chunk_size and current_sentence_chunk:
                        chunks.append(" ".join(current_sentence_chunk).strip())
                        current_sentence_chunk = []
                        current_sentence_size = 0
                    
                    current_sentence_chunk.append(sentence)
                    current_sentence_size += sentence_size
                
                if current_sentence_chunk:
                    chunks.append(" ".join(current_sentence_chunk).strip())
                
                continue  # Skip adding this paragraph to the regular chunks
            
            # If adding this paragraph would exceed the chunk size and we already have content,
            # complete the current chunk and start a new one
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())
                
                # Start new chunk with overlap by including some previous paragraphs
                overlap_size = 0
                overlap_paras = []
                for prev_para in reversed(current_chunk):
                    if overlap_size + len(prev_para) + 2 <= self.chunk_overlap:
                        overlap_paras.insert(0, prev_para)
                        overlap_size += len(prev_para) + 2
                    else:
                        break
                
                current_chunk = overlap_paras
                current_size = overlap_size
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())
        
        return chunks
    
    def chunk_pdf_by_pages(self, pdf_data: Dict[str, Any]) -> List[str]:
        """
        Chunk PDF content by grouping pages together until chunk_size is approached
        This avoids splitting in the middle of pages and provides cleaner chunks
        """
        pages = pdf_data["metadata"].get("pdf_original_pages", [])
        if not pages:
            logger.warning("No PDF pages found for page-based chunking")
            return [pdf_data["text"]]  # Return the full text as one chunk if no pages
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for page in pages:
            page_size = len(page)
            
            # If adding this page would exceed chunk_size and we already have content,
            # complete the current chunk and start a new one
            if current_size + page_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Add the page to the current chunk
            current_chunk.append(page)
            current_size += page_size
        
        # Add any remaining pages
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def remove_duplicate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Remove chunks that are too similar to each other
        """
        unique_chunks = []
        chunk_hashes = set()
        
        for chunk in chunks:
            # Create a simple hash of the first 100 chars to detect very similar chunks
            chunk_start = chunk[:100].strip().lower()
            chunk_hash = hashlib.md5(chunk_start.encode()).hexdigest()
            
            # Only add chunk if we haven't seen this beginning before
            if chunk_hash not in chunk_hashes:
                unique_chunks.append(chunk)
                chunk_hashes.add(chunk_hash)
        
        return unique_chunks

class EmbeddingGenerator:
    """Class for generating text embeddings"""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
    
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using Azure OpenAI service
        """
        if self.config.skip_embeddings or not self.config.endpoint or not self.config.api_key:
            logger.warning("Skipping embeddings generation due to configuration")
            return []
        
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.config.api_key
            }
            
            # Truncate text if it's too long (token limit for embeddings)
            max_length = 8000  # Approximate token limit
            if len(text) > max_length:
                text = text[:max_length]
            
            body = {
                "input": text,
                "model": self.config.deployment_name
            }
            
            url = f"{self.config.endpoint}/openai/deployments/{self.config.deployment_name}/embeddings?api-version=2023-05-15"
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            
            result = response.json()
            embeddings = result["data"][0]["embedding"]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Azure OpenAI: {e}")
            # Return empty embeddings in case of error
            return []

class AzureSearchIndexManager:
    """Class for managing Azure AI Search indexes"""
    
    def __init__(self, config: AzureSearchConfig):
        self.config = config
        self.credential = AzureKeyCredential(config.api_key)
        self.index_client = SearchIndexClient(endpoint=config.endpoint, credential=self.credential)
    
    def create_or_update_index(self) -> None:
        """
        Create or update an Azure AI Search index
        """
        if self.config.force_recreate:
            self._delete_index_if_exists()
        
        try:
            # Create the index
            index = self._build_index_definition()
            
            # Log the index configuration for debugging
            index_dict = index.as_dict()
            logger.info(f"Creating index with configuration: {json.dumps(index_dict, indent=2)[:500]}...")
            
            # Check if semantic settings are present
            if "semantic" in index_dict:
                logger.info("Semantic settings are present in the index configuration")
            else:
                logger.warning("Semantic settings are NOT present in the index configuration")
                
                # If semantic search is enabled but settings are not in the serialized index,
                # we need to manually add them
                if self.config.semantic_search:
                    logger.info("Manually adding semantic settings to the index")
                    
                    # Create a raw index definition that includes semantic settings
                    raw_index = {
                        "name": self.config.index_name,
                        "fields": [
                            {
                                "name": "id",
                                "type": "Edm.String",
                                "key": True,
                                "searchable": False,
                                "filterable": False,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "content",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": False,
                                "sortable": False,
                                "facetable": False,
                                "analyzer": "en.microsoft"
                            },
                            {
                                "name": "file_name",
                                "type": "Edm.String",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "file_path",
                                "type": "Edm.String",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "file_extension",
                                "type": "Edm.String",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "file_size",
                                "type": "Edm.Int64",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "last_modified",
                                "type": "Edm.DateTimeOffset",
                                "searchable": False,
                                "filterable": True,
                                "sortable": True,
                                "facetable": False
                            },
                            {
                                "name": "chunk_id",
                                "type": "Edm.Int32",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "chunk_total",
                                "type": "Edm.Int32",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "language",
                                "type": "Edm.String",
                                "searchable": False,
                                "filterable": True,
                                "sortable": False,
                                "facetable": False
                            },
                            {
                                "name": "title",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": False,
                                "sortable": False,
                                "facetable": False,
                                "analyzer": "en.microsoft"
                            },
                            {
                                "name": "author",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": False,
                                "sortable": False,
                                "facetable": False,
                                "analyzer": "en.microsoft"
                            }
                        ],
                        "semantic": {
                            "configurations": [{
                                "name": "semantic-config",
                                "prioritizedFields": {
                                    "titleField": {"fieldName": "title"},
                                    "prioritizedContentFields": [{"fieldName": "content"}],
                                    "prioritizedKeywordsFields": []
                                }
                            }]
                        }
                    }
                    
                    # Add vector search configuration if needed
                    if self.config.vector_search:
                        raw_index["fields"].append({
                            "name": "content_vector",
                            "type": "Collection(Edm.Single)",
                            "searchable": True,
                            "dimensions": self.config.vector_dimensions,
                            "vectorSearchConfiguration": "vector-config"
                        })
                        
                        raw_index["vectorSearch"] = {
                            "algorithms": [
                                {
                                    "name": "vector-config",
                                    "kind": "hnsw",
                                    "hnswParameters": {
                                        "m": 16,
                                        "efConstruction": 400,
                                        "efSearch": 500,
                                        "metric": "cosine"
                                    }
                                }
                            ]
                        }
                    
                    # Use the REST API directly
                    # Use the correct API version for semantic search
                    api_version = "2024-11-01-preview"
                    endpoint = f"{self.config.endpoint}/indexes/{self.config.index_name}?api-version={api_version}"
                    headers = {
                        "Content-Type": "application/json",
                        "api-key": self.config.api_key
                    }
                    
                    logger.info(f"Using REST API with API version: {api_version}")
                    
                    try:
                        # Send the request
                        response = requests.put(endpoint, headers=headers, json=raw_index)
                        response_text = response.text
                        
                        if response.status_code >= 400:
                            logger.error(f"Error response: {response_text}")
                            response.raise_for_status()
                        
                        logger.info(f"Index {self.config.index_name} created with semantic settings via REST API")
                        return
                    except Exception as custom_request_error:
                        logger.error(f"Error with REST API request: {custom_request_error}")
                        logger.info("Falling back to SDK method without semantic settings")
            
            # Use the SDK to create the index if we didn't use the REST API
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Index {self.config.index_name} created or updated successfully")
        except Exception as e:
            logger.error(f"Error creating index {self.config.index_name}: {e}")
            logger.info("Continuing with existing index configuration...")
    
    def _delete_index_if_exists(self) -> None:
        """Delete the index if it exists"""
        try:
            logger.info(f"Attempting to delete index {self.config.index_name} for recreation")
            self.index_client.delete_index(self.config.index_name)
            logger.info(f"Successfully deleted index {self.config.index_name}")
        except Exception as e:
            logger.info(f"Index deletion unsuccessful (may not exist): {e}")
    
    def _build_index_definition(self) -> SearchIndex:
        """
        Build the search index definition
        """
        # Import basic models
        from azure.search.documents.indexes.models import (
            SearchableField, 
            SimpleField, 
            SearchFieldDataType,
            SearchIndex
        )
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="file_extension", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="file_size", type=SearchFieldDataType.Int64, filterable=True),
            SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="chunk_total", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="language", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="author", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        ]
        
        # Add vector search capability if requested
        index_additional_properties = {}
        
        if self.config.vector_search:
            logger.info("Adding vector search configuration to index")
            # Create a field definition manually with required vector properties
            content_vector_field = {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": self.config.vector_dimensions,
                "vectorSearchConfiguration": "vector-config"
            }
            
            # Set up vector search configuration
            vector_search_config = {
                "algorithms": [
                    {
                        "name": "vector-config",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "m": 16,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    }
                ]
            }
            
            # Add to index properties
            index_additional_properties["vectorSearch"] = vector_search_config
        
        if self.config.semantic_search:
            logger.info("Adding semantic search configuration to index")
            # Use a dictionary-based configuration for semantic search
            semantic_config = {
                "configurations": [{
                    "name": "semantic-config",
                    "prioritizedFields": {
                        "titleField": {"fieldName": "title"},
                        "prioritizedContentFields": [{"fieldName": "content"}],
                        "prioritizedKeywordsFields": []
                    }
                }]
            }
            
            index_additional_properties["semantic"] = semantic_config
            logger.info(f"Semantic configuration: {json.dumps(semantic_config, indent=2)}")
        
        # Create the index with basic fields
        index = SearchIndex(name=self.config.index_name, fields=fields)
        
        # Add any additional properties to the index
        for key, value in index_additional_properties.items():
            logger.info(f"Adding additional property to index: {key}")
            index.additional_properties[key] = value
        
        # For vector search, we need to hack around SDK limitations by modifying the serialized index
        if self.config.vector_search:
            logger.info("Applying vector search workaround")
            # Convert the index to a dictionary for direct manipulation
            index_dict = index.as_dict()
            
            # Get the current fields list
            current_fields = index_dict.get("fields", [])
            
            # Add our custom vector field directly
            current_fields.append(content_vector_field)
            
            # Create a new index with our modified properties
            new_index = SearchIndex(name=self.config.index_name)
            
            # Copy all properties from the original index dictionary
            for key, value in index_dict.items():
                if key == "fields":
                    # Skip fields as we'll handle them separately
                    continue
                logger.info(f"Copying property to new index: {key}")
                new_index.additional_properties[key] = value
            
            # Add our modified fields
            new_index.additional_properties["fields"] = current_fields
            
            # Make sure semantic settings are preserved if enabled
            if self.config.semantic_search and "semantic" in index_additional_properties:
                logger.info("Explicitly adding semantic settings to new index")
                new_index.additional_properties["semantic"] = index_additional_properties["semantic"]
            
            # Check if semantic settings are in the new index
            if "semantic" in new_index.additional_properties:
                logger.info("Semantic settings are present in the new index")
            else:
                logger.warning("Semantic settings are NOT present in the new index")
            
            # Use the new index with properly configured fields and settings
            return new_index
        
        return index

class DocumentUploader:
    """Class for uploading documents to Azure AI Search"""
    
    def __init__(
        self, 
        azure_config: AzureSearchConfig,
        openai_config: OpenAIConfig
    ):
        self.azure_config = azure_config
        self.openai_config = openai_config
        
        self.credential = AzureKeyCredential(azure_config.api_key)
        self.search_client = SearchClient(
            endpoint=azure_config.endpoint, 
            index_name=azure_config.index_name, 
            credential=self.credential
        )
    
    def upload_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Upload document chunks to Azure AI Search index
        """
        try:
            # If we're doing vector search but the index doesn't have the vector field properly set up
            # We should filter out the vector embeddings
            if self.azure_config.vector_search:
                try:
                    # Try to verify the index has a content_vector field
                    index_client = SearchIndexClient(
                        endpoint=self.search_client._endpoint, 
                        credential=self.search_client._credential
                    )
                    index = index_client.get_index(self.search_client._index_name)
                    
                    has_vector_field = False
                    for field in index.fields:
                        if field.name == "content_vector":
                            has_vector_field = True
                            break
                    
                    if not has_vector_field:
                        logger.warning("Index does not contain a properly configured content_vector field. Removing vector data from chunks.")
                        for chunk in chunks:
                            if "content_vector" in chunk:
                                del chunk["content_vector"]
                except Exception as e:
                    logger.warning(f"Could not verify index structure: {e}. Attempting upload anyway.")
                    
            # Upload in batches of 1000 (Azure Search limit)
            batch_size = 1000
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Make sure all documents have valid data types
                batch = [self._ensure_valid_document(doc) for doc in batch]
                
                # Log a small preview of what we're sending
                if i == 0:
                    # Create a deep copy of the first document for logging (to avoid modifying the original)
                    sample_doc = dict(batch[0])
                    # Remove vector field for logging if it exists (it's too large to print)
                    if "content_vector" in sample_doc:
                        sample_doc["content_vector"] = "[vector data]"
                    logger.info(f"Sample document being uploaded: {json.dumps(sample_doc, indent=2)[:500]}...")
                
                try:
                    result = self.search_client.upload_documents(documents=batch)
                    logger.info(f"Uploaded batch of {len(batch)} chunks (from {i} to {i+len(batch)-1})")
                    success_count = sum(1 for r in result if r.succeeded)
                    logger.info(f"Upload results: {success_count}/{len(batch)} succeeded")
                    
                    # If there were failures, log details about the first few
                    if success_count < len(batch):
                        failed_count = len(batch) - success_count
                        logger.warning(f"{failed_count} document uploads failed")
                        for j, r in enumerate(result):
                            if not r.succeeded and j < 5:  # Log up to 5 failures
                                logger.warning(f"Document {j} failed: {r.error_message}")
                
                except Exception as batch_error:
                    logger.error(f"Error uploading batch {i//batch_size + 1}: {batch_error}")
                    
                    # Try without content_vector field if that's the issue
                    if "content_vector" in str(batch_error).lower():
                        logger.info("Trying upload without vector embeddings...")
                        try:
                            for doc in batch:
                                if "content_vector" in doc:
                                    del doc["content_vector"]
                            result = self.search_client.upload_documents(documents=batch)
                            logger.info(f"Upload without vectors succeeded: {sum(1 for r in result if r.succeeded)}/{len(batch)}")
                        except Exception as retry_error:
                            logger.error(f"Failed even without vectors: {retry_error}")
                            raise
                    else:
                        raise
                    
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise
    
    def _ensure_valid_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all fields have valid data types for Azure Search
        """
        # Current time in ISO 8601 UTC format as a fallback
        default_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Fields that are expected to be dates
        date_fields = ["last_modified"]
        
        # Check and fix date fields
        for field in date_fields:
            if field in document:
                if not document[field] or not isinstance(document[field], str):
                    logger.warning(f"Invalid date value for {field}, using default date")
                    document[field] = default_date
                elif not (document[field].endswith('Z') or '+' in document[field]):
                    # Add Z for UTC if no timezone info
                    document[field] = document[field].rstrip() + 'Z'
        
        return document

class RAGDocumentProcessor:
    """
    Main class that orchestrates the entire process of processing documents for RAG
    """
    
    def __init__(
        self,
        processing_config: ProcessingConfig,
        azure_config: AzureSearchConfig,
        openai_config: OpenAIConfig
    ):
        self.processing_config = processing_config
        self.azure_config = azure_config
        self.openai_config = openai_config
        
        self.document_processor = DocumentProcessor(processing_config)
        self.text_chunker = TextChunker(processing_config.chunk_size, processing_config.chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(openai_config)
        self.index_manager = AzureSearchIndexManager(azure_config)
        self.document_uploader = DocumentUploader(azure_config, openai_config)
    
    def process(self) -> None:
        """
        Process all documents and upload to Azure AI Search
        """
        # Create or update the search index
        self.index_manager.create_or_update_index()
        
        # Process files and upload chunks
        all_chunks = []
        
        # Walk through the directory and process each file
        for root, _, files in os.walk(self.processing_config.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file matches include filters or should be excluded
                if not self._should_process_file(file_path):
                    logger.info(f"Skipping file based on filters: {file_path}")
                    continue
                
                logger.info(f"Processing file: {file_path}")
                
                # Extract text and metadata from file
                result = self.document_processor.process_file(file_path)
                text = result["text"]
                metadata = result["metadata"]
                
                if not text:
                    logger.warning(f"No text content extracted from {file_path}")
                    continue
                
                # Detect language
                language = self._detect_language(text)
                
                # Use standard chunking for all file types now
                text_chunks = self.text_chunker.chunk_text(text)
                logger.info(f"Split {file_path} into {len(text_chunks)} semantic chunks")
                
                # Remove duplicate chunks
                unique_chunks = self.text_chunker.remove_duplicate_chunks(text_chunks)
                if len(unique_chunks) < len(text_chunks):
                    logger.info(f"Removed {len(text_chunks) - len(unique_chunks)} duplicate chunks")
                
                # Current time in ISO 8601 format for any missing dates
                default_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Create documents for each unique chunk
                for i, chunk in enumerate(unique_chunks):
                    # Generate a stable ID based on file path and chunk index
                    chunk_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()
                    
                    # Create title with page information if available
                    # Extract page numbers from the chunk if present
                    page_numbers = []
                    if file_path.lower().endswith('.pdf'):
                        page_markers = re.findall(r'\[pg:(\d+)\]', chunk)
                        if page_markers:
                            page_numbers = sorted(set(int(p) for p in page_markers))
                            
                    # Create a descriptive title
                    if page_numbers:
                        if len(page_numbers) == 1:
                            chunk_title = f"{os.path.basename(file_path)} (Page {page_numbers[0]})"
                        else:
                            chunk_title = f"{os.path.basename(file_path)} (Pages {min(page_numbers)}-{max(page_numbers)})"
                    else:
                        chunk_title = metadata.get("title", os.path.basename(file_path))
                        
                        # Try to extract a better title from the first line
                        first_line = chunk.split('\n', 1)[0].strip()
                        if first_line and len(first_line) < 100 and not first_line.startswith('[') and not first_line.startswith('#'):
                            chunk_title = f"{os.path.basename(file_path)}: {first_line}"
                    
                    # Ensure we have a valid last_modified date
                    last_modified = metadata.get("last_modified")
                    if not last_modified or last_modified == "":
                        last_modified = default_date
                    
                    document = {
                        "id": chunk_id,
                        "content": chunk,
                        "file_name": metadata.get("file_name", ""),
                        "file_path": metadata.get("file_path", ""),
                        "file_extension": metadata.get("file_extension", ""),
                        "file_size": metadata.get("file_size", 0),
                        "last_modified": last_modified,
                        "chunk_id": i,
                        "chunk_total": len(unique_chunks),
                        "language": language,
                        "title": chunk_title,
                        "author": metadata.get("author", "")
                    }
                    
                    # Add embeddings if vector search is enabled
                    if self.azure_config.vector_search and not self.openai_config.skip_embeddings:
                        try:
                            logger.info(f"Generating embeddings for chunk {i+1}/{len(unique_chunks)} of {file_path}")
                            embeddings = self.embedding_generator.generate_embeddings(chunk)
                            if embeddings:
                                document["content_vector"] = embeddings
                            # Add a short delay to avoid rate limiting
                            time.sleep(0.1)
                        except Exception as e:
                            logger.error(f"Error generating embeddings: {e}")
                            logger.info("Continuing without embeddings for this chunk...")
                    
                    all_chunks.append(document)
        
        # Upload chunks to the search index
        if all_chunks:
            logger.info(f"Uploading {len(all_chunks)} chunks to index {self.azure_config.index_name}")
            self.document_uploader.upload_chunks(all_chunks)
            logger.info("Upload completed successfully")
        else:
            logger.warning("No content was extracted from the files. Nothing to upload.")
    
    def _should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on include/exclude filters
        """
        # If no filters are set, process all files
        if not self.processing_config.include_filters and not self.processing_config.exclude_filters:
            return True
        
        file_name = os.path.basename(file_path).lower()
        
        # Check exclude filters first
        for pattern in self.processing_config.exclude_filters:
            if re.search(pattern.lower(), file_name) or re.search(pattern.lower(), file_path.lower()):
                return False
        
        # If include filters are specified, at least one must match
        if self.processing_config.include_filters:
            for pattern in self.processing_config.include_filters:
                if re.search(pattern.lower(), file_name) or re.search(pattern.lower(), file_path.lower()):
                    return True
            # If we get here, no include filter matched
            return False
        
        # No include filters and excluded passed
        return True
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text
        """
        try:
            # Use a sample of the text for faster detection
            sample = text[:min(1000, len(text))]
            return detect(sample)
        except:
            # Default to English if detection fails
            return "en"

def main():
    """
    Main entry point for the script
    """
    parser = argparse.ArgumentParser(description='Upload documents to Azure AI Search for RAG')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing documents to upload')
    parser.add_argument('--endpoint', type=str, required=True, help='Azure AI Search endpoint')
    parser.add_argument('--key', type=str, required=True, help='Azure AI Search admin key')
    parser.add_argument('--index', type=str, required=True, help='Name of the search index')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of text chunks (in characters)')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Overlap between chunks (in characters)')
    parser.add_argument('--vector-search', action='store_true', help='Enable vector search capabilities')
    parser.add_argument('--semantic-search', action='store_true', help='Enable semantic search capabilities')
    parser.add_argument('--aoai-endpoint', type=str, help='Azure OpenAI endpoint for embeddings')
    parser.add_argument('--aoai-key', type=str, help='Azure OpenAI key for embeddings')
    parser.add_argument('--aoai-deployment', type=str, help='Azure OpenAI deployment name for embeddings model')
    parser.add_argument('--vector-dimensions', type=int, default=1536, help='Dimensions for vector embeddings')
    parser.add_argument('--skip-vectors', action='store_true', help='Skip vector processing even if --vector-search is enabled')
    parser.add_argument('--force-recreate', action='store_true', help='Force recreation of the index')
    parser.add_argument('--include', type=str, nargs='+', default=[], help='File patterns to include (regex)')
    parser.add_argument('--exclude', type=str, nargs='+', default=[], help='File patterns to exclude (regex)')
    
    args = parser.parse_args()
    
    # Validate vector search parameters
    if args.vector_search and not args.skip_vectors and (not args.aoai_endpoint or not args.aoai_key or not args.aoai_deployment):
        logger.error("Vector search requires Azure OpenAI parameters: --aoai-endpoint, --aoai-key, and --aoai-deployment")
        return
    
    # Create configuration objects
    processing_config = ProcessingConfig(
        folder_path=args.folder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_filters=args.include,
        exclude_filters=args.exclude
    )
    
    azure_config = AzureSearchConfig(
        endpoint=args.endpoint,
        api_key=args.key,
        index_name=args.index,
        vector_search=args.vector_search,
        semantic_search=args.semantic_search,
        vector_dimensions=args.vector_dimensions,
        force_recreate=args.force_recreate
    )
    
    openai_config = OpenAIConfig(
        endpoint=args.aoai_endpoint,
        api_key=args.aoai_key,
        deployment_name=args.aoai_deployment,
        skip_embeddings=args.skip_vectors
    )
    
    # Create and run the document processor
    processor = RAGDocumentProcessor(processing_config, azure_config, openai_config)
    processor.process()

if __name__ == "__main__":
    main()