import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import json
import time
from collections import deque
from typing import List, Dict, Set, Tuple
import pandas as pd
from datetime import datetime


class WebCrawler:
    # A single edited word invalidates ``shingle_size`` shingles, so on
    # short documents even a two-word paraphrase drops Jaccard similarity to ~
    # 0.79.
    # 0.75 is the usual operating point for near-duplicate detection and is
    # still far above the similarity of genuinely distinct documents (< 0.1).
    def __init__(self, shingle_size: int = 5, near_duplicate_threshold: float = 0.75):
        self.visited_urls: Set[str] = set()
        self.document_hashes: Set[str] = set()
        self.documents: List[Dict] = []
        self.metadata: List[Dict] = []
        self.url_graph: Dict[str, List[str]] = {}
        self.shingle_size = shingle_size
        self.near_duplicate_threshold = near_duplicate_threshold
        self.document_shingles: List[Set[int]] = []
        self.duplicate_report: List[Dict] = []
        self.last_stats: Dict = {}
        
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
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str,
                        base_domain: str = None) -> List[str]:
        """Extract all links from a page, honouring the domain restriction."""
        links = []
        seen = set()
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(base_url, link['href'])
            # base_domain must be forwarded here, otherwise the
            # "stay on domain" option has no effect at all.
            if self._is_valid_url(absolute_url, base_domain):
                normalized = self._normalize_url(absolute_url)
                if normalized not in seen:
                    seen.add(normalized)
                    links.append(normalized)
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
        """Compute hash of content for exact duplicate detection."""
        return hashlib.md5(content.encode()).hexdigest()

    def _compute_shingles(self, content: str) -> Set[int]:
        """Compute the set of hashed word w-shingles for a document.

        Exact hashing only catches byte-identical pages. Shingling captures
        local word order, so lightly edited mirrors (different boilerplate,
        a changed sentence) still overlap heavily and can be detected with
        Jaccard similarity.
        """
        words = content.lower().split()
        size = self.shingle_size
        if len(words) < size:
            return {hash(' '.join(words))} if words else set()
        return {
            hash(' '.join(words[i:i + size]))
            for i in range(len(words) - size + 1)
        }

    @staticmethod
    def _jaccard(left: Set[int], right: Set[int]) -> float:
        """Jaccard similarity between two shingle sets."""
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        return intersection / union if union else 0.0

    def find_near_duplicate(self, content: str) -> Tuple[int, float]:
        """Return (index, similarity) of the closest stored near-duplicate.

        Returns ``(-1, best_similarity)`` when nothing crosses the threshold.
        """
        shingles = self._compute_shingles(content)
        best_index, best_similarity = -1, 0.0

        for index, existing in enumerate(self.document_shingles):
            similarity = self._jaccard(shingles, existing)
            if similarity > best_similarity:
                best_index, best_similarity = index, similarity

        if best_similarity >= self.near_duplicate_threshold:
            return best_index, best_similarity
        return -1, best_similarity
    
    def crawl(self, seed_urls: List[str], max_depth: int = 2, max_pages: int = 100, 
              stay_on_domain: bool = True, delay: float = 0.5,
              detect_near_duplicates: bool = True) -> Dict:
        """
        Crawl web pages starting from seed URLs.
        
        Args:
            seed_urls: List of starting URLs
            max_depth: Maximum depth to crawl
            max_pages: Maximum number of pages to crawl
            stay_on_domain: Whether to stay on the domains of the seed URLs
            delay: Politeness delay between requests, in seconds
            detect_near_duplicates: Enable shingle-based near-duplicate removal
        
        Returns:
            Dictionary with crawl statistics
        """
        started = time.perf_counter()

        self.visited_urls.clear()
        self.document_hashes.clear()
        self.documents.clear()
        self.metadata.clear()
        self.url_graph.clear()
        self.document_shingles.clear()
        self.duplicate_report.clear()
        
        queue = deque()
        seed_domains = set()
        urls_discovered = 0
        
        # Initialize queue with seed URLs. Every seed contributes its domain,
        # so multiple heterogeneous seed sources are supported.
        for seed_url in seed_urls:
            if self._is_valid_url(seed_url):
                normalized_url = self._normalize_url(seed_url)
                queue.append((normalized_url, 0))
                urls_discovered += 1
                seed_domains.add(self._get_domain(normalized_url))
        
        pages_crawled = 0
        duplicate_urls = 0
        depth_limited = 0
        exact_duplicates = 0
        near_duplicates = 0
        fetch_errors = 0
        
        while queue and pages_crawled < max_pages:
            url, depth = queue.popleft()
            
            # Distinct counters: an already-visited URL is a duplicate URL,
            # whereas a URL beyond max_depth was simply not explored.
            if url in self.visited_urls:
                duplicate_urls += 1
                continue
            if depth > max_depth:
                depth_limited += 1
                continue
            
            self.visited_urls.add(url)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    fetch_errors += 1
                    continue
                
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Extract content and metadata
                content = self._extract_content(soup)
                metadata = self._extract_metadata(soup, url)
                
                if not content.strip():
                    fetch_errors += 1
                    continue
                
                # Exact duplicate detection via content hash
                content_hash = self._compute_hash(content)
                if content_hash in self.document_hashes:
                    exact_duplicates += 1
                    self.duplicate_report.append({
                        'url': url, 'type': 'exact duplicate',
                        'similarity': 1.0, 'matched_document': None
                    })
                    continue
                
                # Near-duplicate detection via shingle Jaccard similarity
                if detect_near_duplicates:
                    match_index, similarity = self.find_near_duplicate(content)
                    if match_index >= 0:
                        near_duplicates += 1
                        self.duplicate_report.append({
                            'url': url, 'type': 'near duplicate',
                            'similarity': round(similarity, 3),
                            'matched_document': self.documents[match_index]['url']
                        })
                        continue
                
                self.document_hashes.add(content_hash)
                self.document_shingles.append(self._compute_shingles(content))
                
                # Store document content separately from its metadata
                self.documents.append({
                    'doc_id': len(self.documents),
                    'url': url,
                    'content': content,
                    'hash': content_hash
                })
                
                metadata['doc_id'] = len(self.documents) - 1
                metadata['depth'] = depth
                metadata['word_count'] = len(content.split())
                self.metadata.append(metadata)
                
                # Extract links and build graph (domain restriction applied)
                links = self._extract_links(soup, url, base_domain=None if not stay_on_domain else self._get_domain(url))
                if stay_on_domain:
                    links = [link for link in links if self._get_domain(link) in seed_domains]
                self.url_graph[url] = links
                
                # Add new links to queue
                for link in links:
                    if link not in self.visited_urls:
                        queue.append((link, depth + 1))
                        urls_discovered += 1
                
                pages_crawled += 1
                if delay:
                    time.sleep(delay)  # Be polite to servers
                
            except Exception as e:
                fetch_errors += 1
                print(f"Error crawling {url}: {e}")
                continue
        
        elapsed = time.perf_counter() - started
        self.last_stats = {
            'pages_crawled': pages_crawled,
            'seed_urls': len(seed_urls),
            'seed_domains': sorted(seed_domains),
            'max_depth': max_depth,
            'urls_discovered': urls_discovered,
            'unique_urls_visited': len(self.visited_urls),
            'duplicate_urls_skipped': duplicate_urls,
            'depth_limited_urls': depth_limited,
            'duplicate_documents_skipped': exact_duplicates,
            'near_duplicate_documents_skipped': near_duplicates,
            'fetch_errors': fetch_errors,
            'graph_edges': sum(len(links) for links in self.url_graph.values()),
            'crawl_seconds': round(elapsed, 2),
            'pages_per_second': round(pages_crawled / elapsed, 2) if elapsed > 0 else 0
        }
        return self.last_stats
    
    def add_documents(self, documents: List[Dict], metadata: List[Dict] = None,
                        detect_near_duplicates: bool = True) -> Dict:
        """Ingest documents from a non-crawled source (dataset, API, upload).

        Applies the same duplicate and near-duplicate policy as the crawler so
        that heterogeneous sources can be combined into one clean collection.
        """
        added = 0
        exact_duplicates = 0
        near_duplicates = 0

        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            if not content.strip():
                continue

            url = doc.get('url', f'local://document/{i}')
            if url in self.visited_urls:
                continue

            content_hash = doc.get('hash') or self._compute_hash(content)
            if content_hash in self.document_hashes:
                exact_duplicates += 1
                self.duplicate_report.append({
                    'url': url, 'type': 'exact duplicate',
                    'similarity': 1.0, 'matched_document': None
                })
                continue

            if detect_near_duplicates:
                match_index, similarity = self.find_near_duplicate(content)
                if match_index >= 0:
                    near_duplicates += 1
                    self.duplicate_report.append({
                        'url': url, 'type': 'near duplicate',
                        'similarity': round(similarity, 3),
                        'matched_document': self.documents[match_index]['url']
                    })
                    continue

            self.visited_urls.add(url)
            self.document_hashes.add(content_hash)
            self.document_shingles.append(self._compute_shingles(content))
            self.documents.append({
                'doc_id': len(self.documents),
                'url': url,
                'content': content,
                'hash': content_hash
            })

            record = dict(metadata[i]) if metadata and i < len(metadata) else {'url': url}
            record.setdefault('url', url)
            record['doc_id'] = len(self.documents) - 1
            record.setdefault('word_count', len(content.split()))
            record.setdefault('crawl_date', datetime.now().isoformat(timespec='seconds'))
            self.metadata.append(record)
            added += 1

        return {
            'documents_added': added,
            'duplicate_documents_skipped': exact_duplicates,
            'near_duplicate_documents_skipped': near_duplicates,
            'total_documents': len(self.documents)
        }

    def get_duplicate_report(self) -> pd.DataFrame:
        """Return the duplicate / near-duplicate decisions as a table."""
        if not self.duplicate_report:
            return pd.DataFrame(columns=['url', 'type', 'similarity', 'matched_document'])
        return pd.DataFrame(self.duplicate_report)
    
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
