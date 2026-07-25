import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import json
import time
from collections import deque
from typing import List, Dict, Set
import pandas as pd
from datetime import datetime


class WebCrawler:
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.document_hashes: Set[str] = set()
        self.documents: List[Dict] = []
        self.metadata: List[Dict] = []
        self.url_graph: Dict[str, List[str]] = {}
        
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _is_valid_url(self, url: str, base_domain: str = None) -> bool:
        """Check if URL is valid and should be crawled."""
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            if parsed.scheme not in ['http', 'https']:
                return False
            if base_domain and self._get_domain(url) != base_domain:
                return False
            # Skip common non-content URLs
            excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe']
            if any(url.lower().endswith(ext) for ext in excluded_extensions):
                return False
            return True
        except:
            return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        normalized = parsed._replace(fragment='', params='')
        if normalized.path.endswith('/'):
            normalized = normalized._replace(path=normalized.path.rstrip('/'))
        return normalized.geturl()
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from a page."""
        links = []
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(base_url, link['href'])
            if self._is_valid_url(absolute_url):
                links.append(self._normalize_url(absolute_url))
        return links
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text from main content areas
        text = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']):
            text += tag.get_text() + " "
        
        return ' '.join(text.split())
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'description': '',
            'keywords': '',
            'crawl_date': datetime.now().isoformat()
        }
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Extract h1
        h1_tags = soup.find_all('h1')
        metadata['headings'] = [h1.get_text() for h1 in h1_tags]
        
        return metadata
    
    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for duplicate detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def crawl(self, seed_urls: List[str], max_depth: int = 2, max_pages: int = 100, 
              stay_on_domain: bool = True) -> Dict:
        """
        Crawl web pages starting from seed URLs.
        
        Args:
            seed_urls: List of starting URLs
            max_depth: Maximum depth to crawl
            max_pages: Maximum number of pages to crawl
            stay_on_domain: Whether to stay on the same domain as seed URLs
        
        Returns:
            Dictionary with crawl statistics
        """
        self.visited_urls.clear()
        self.document_hashes.clear()
        self.documents.clear()
        self.metadata.clear()
        self.url_graph.clear()
        
        queue = deque()
        base_domain = None
        
        # Initialize queue with seed URLs
        for seed_url in seed_urls:
            if self._is_valid_url(seed_url):
                normalized_url = self._normalize_url(seed_url)
                queue.append((normalized_url, 0))
                if stay_on_domain and base_domain is None:
                    base_domain = self._get_domain(normalized_url)
        
        pages_crawled = 0
        duplicate_urls = 0
        duplicate_documents = 0
        
        while queue and pages_crawled < max_pages:
            url, depth = queue.popleft()
            
            if url in self.visited_urls or depth > max_depth:
                duplicate_urls += 1
                continue
            
            self.visited_urls.add(url)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Extract content and metadata
                content = self._extract_content(soup)
                metadata = self._extract_metadata(soup, url)
                
                # Check for duplicate content
                content_hash = self._compute_hash(content)
                if content_hash in self.document_hashes:
                    duplicate_documents += 1
                    continue
                
                self.document_hashes.add(content_hash)
                
                # Store document
                self.documents.append({
                    'url': url,
                    'content': content,
                    'hash': content_hash
                })
                
                # Store metadata
                self.metadata.append(metadata)
                
                # Extract links and build graph
                links = self._extract_links(soup, url)
                self.url_graph[url] = links
                
                # Add new links to queue
                for link in links:
                    if link not in self.visited_urls:
                        queue.append((link, depth + 1))
                
                pages_crawled += 1
                time.sleep(0.5)  # Be polite to servers
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                continue
        
        return {
            'pages_crawled': pages_crawled,
            'duplicate_urls_skipped': duplicate_urls,
            'duplicate_documents_skipped': duplicate_documents,
            'total_urls_found': len(self.visited_urls),
            'graph_edges': sum(len(links) for links in self.url_graph.values())
        }
    
    def save_documents(self, filepath: str):
        """Save crawled documents to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def save_metadata(self, filepath: str):
        """Save metadata to CSV file."""
        df = pd.DataFrame(self.metadata)
        df.to_csv(filepath, index=False)
    
    def save_graph(self, filepath: str):
        """Save URL graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.url_graph, f, ensure_ascii=False, indent=2)
    
    def load_documents(self, filepath: str):
        """Load documents from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
    
    def load_metadata(self, filepath: str):
        """Load metadata from CSV file."""
        self.metadata = pd.read_csv(filepath).to_dict('records')
    
    def load_graph(self, filepath: str):
        """Load URL graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.url_graph = json.load(f)
    
    def get_documents_dataframe(self) -> pd.DataFrame:
        """Return documents as pandas DataFrame."""
        return pd.DataFrame(self.documents)
    
    def get_metadata_dataframe(self) -> pd.DataFrame:
        """Return metadata as pandas DataFrame."""
        return pd.DataFrame(self.metadata)
