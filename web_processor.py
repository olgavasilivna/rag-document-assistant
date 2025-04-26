from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from urllib.parse import urlparse, urljoin, unquote
import logging

class WebProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _extract_actual_url(self, url: str) -> str:
        """Extract the actual URL from a redirect URL."""
        try:
            # Handle Google redirect URLs
            if 'google.com/url' in url:
                # Extract the actual URL from the 'url' parameter
                parsed = urlparse(url)
                query_params = dict(param.split('=') for param in parsed.query.split('&'))
                if 'url' in query_params:
                    return unquote(query_params['url'])
            
            # Follow redirects to get the final URL
            response = self.session.head(url, allow_redirects=True)
            return response.url
        except Exception as e:
            logging.error(f"Error extracting actual URL: {str(e)}")
            return url
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL."""
        try:
            # Get the actual URL after handling redirects
            actual_url = self._extract_actual_url(url)
            
            # Fetch the webpage content
            response = self.session.get(actual_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    def process_url(self, url: str) -> List[Dict]:
        """Process a URL and return chunks with metadata."""
        try:
            # Extract text from URL
            text = self.extract_text_from_url(url)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": url,
                        "chunk": i + 1
                    }
                })
            
            return processed_chunks
        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            return [{
                "text": f"Error processing URL: {str(e)}",
                "metadata": {"source": url, "error": str(e)}
            }]
    
    def process_urls(self, urls: List[str]) -> List[Dict]:
        """Process multiple URLs and return their contents and metadata."""
        all_chunks = []
        for url in urls:
            chunks = self.process_url(url)
            all_chunks.extend(chunks)
        return all_chunks 