import numpy as np
import networkx as nx
from whoosh.index import create_in, exists_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Or, Term
from whoosh import scoring
import os
import json
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px


class SearchEngine:
    def __init__(self, index_dir: str = "index_dir"):
        self.index_dir = index_dir
        self.index = None
        self.searcher = None
        self.documents = []
        self.url_graph = {}
        self.pagerank_scores = {}
        self.hits_scores = {}
        self.document_vectors = None
        self.tfidf_matrix = None
        
    def create_index(self, documents: List[Dict], metadata: List[Dict] = None):
        """Create Whoosh index from documents."""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        # Don't store full content to avoid serialization issues
        schema = Schema(
            url=ID(stored=True),
            title=TEXT(stored=True),
            content=TEXT(stored=False),  # Index but don't store
            doc_id=ID(stored=True)
        )
        
        # Always create fresh index to avoid conflicts
        if exists_in(self.index_dir):
            # Remove existing index
            import shutil
            shutil.rmtree(self.index_dir)
            os.makedirs(self.index_dir)
        
        self.index = create_in(self.index_dir, schema)
        
        writer = self.index.writer()
        
        for i, doc in enumerate(documents):
            title = metadata[i].get('title', '') if metadata and i < len(metadata) else ''
            # Limit content size to avoid issues
            content = doc['content'][:100000] if len(doc['content']) > 100000 else doc['content']
            
            writer.add_document(
                url=doc['url'],
                title=title[:1000] if len(title) > 1000 else title,
                content=content,
                doc_id=str(i)
            )
        
        writer.commit()
        self.documents = documents
        self.searcher = self.index.searcher()
        
    def build_graph(self, url_graph: Dict[str, List[str]]):
        """Build graph structure from URL links."""
        self.url_graph = url_graph
        G = nx.DiGraph()
        
        for url, links in url_graph.items():
            G.add_node(url)
            for link in links:
                G.add_edge(url, link)
        
        return G
    
    def calculate_pagerank(self, G: nx.DiGraph, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """Calculate PageRank scores."""
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
        self.pagerank_scores = pagerank
        return pagerank
    
    def calculate_hits(self, G: nx.DiGraph, max_iter: int = 100) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate HITS scores (hubs and authorities)."""
        hubs, authorities = nx.hits(G, max_iter=max_iter)
        self.hits_scores = {'hubs': hubs, 'authorities': authorities}
        return hubs, authorities
    
    def search(self, query: str, limit: int = 10, ranking_method: str = 'tfidf') -> List[Dict]:
        """
        Search documents with specified ranking method.
        
        Args:
            query: Search query
            limit: Number of results to return
            ranking_method: 'tfidf', 'pagerank', or 'hits'
        """
        if self.searcher is None:
            self.searcher = self.index.searcher()
        
        query_parser = MultifieldParser(['title', 'content'], self.index.schema)
        parsed_query = query_parser.parse(query)
        
        if ranking_method == 'pagerank':
            # Use custom weighting with PageRank
            results = self.searcher.search(
                parsed_query, 
                limit=limit * 2,  # Get more results for re-ranking
                scored=True
            )
            ranked_results = self._rerank_with_pagerank(results, limit)
        elif ranking_method == 'hits':
            # Use custom weighting with HITS
            results = self.searcher.search(
                parsed_query,
                limit=limit * 2,
                scored=True
            )
            ranked_results = self._rerank_with_hits(results, limit)
        else:
            # Standard TF-IDF ranking
            results = self.searcher.search(parsed_query, limit=limit, scored=True)
            ranked_results = self._format_results(results)
        
        return ranked_results
    
    def _rerank_with_pagerank(self, results, limit: int) -> List[Dict]:
        """Re-rank results using PageRank scores."""
        reranked = []
        for hit in results:
            url = hit['url']
            base_score = hit.score
            pr_score = self.pagerank_scores.get(url, 0)
            combined_score = 0.7 * base_score + 0.3 * pr_score
            
            # Get content from original documents
            doc = next((d for d in self.documents if d['url'] == url), None)
            content = doc['content'][:500] + '...' if doc and len(doc['content']) > 500 else (doc['content'] if doc else '')
            
            reranked.append({
                'url': url,
                'title': hit.get('title', ''),
                'score': combined_score,
                'original_score': base_score,
                'pagerank_score': pr_score,
                'content': content
            })
        
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked[:limit]
    
    def _rerank_with_hits(self, results, limit: int) -> List[Dict]:
        """Re-rank results using HITS authority scores."""
        reranked = []
        authorities = self.hits_scores.get('authorities', {})
        
        for hit in results:
            url = hit['url']
            base_score = hit.score
            auth_score = authorities.get(url, 0)
            combined_score = 0.7 * base_score + 0.3 * auth_score
            
            # Get content from original documents
            doc = next((d for d in self.documents if d['url'] == url), None)
            content = doc['content'][:500] + '...' if doc and len(doc['content']) > 500 else (doc['content'] if doc else '')
            
            reranked.append({
                'url': url,
                'title': hit.get('title', ''),
                'score': combined_score,
                'original_score': base_score,
                'authority_score': auth_score,
                'content': content
            })
        
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked[:limit]
    
    def _format_results(self, results) -> List[Dict]:
        """Format standard search results."""
        formatted = []
        for hit in results:
            url = hit['url']
            
            # Get content from original documents
            doc = next((d for d in self.documents if d['url'] == url), None)
            content = doc['content'][:500] + '...' if doc and len(doc['content']) > 500 else (doc['content'] if doc else '')
            
            formatted.append({
                'url': url,
                'title': hit.get('title', ''),
                'score': hit.score,
                'content': content
            })
        return formatted
    
    def advanced_search(self, query: str, filters: Dict = None, limit: int = 10) -> List[Dict]:
        """Advanced search with filters."""
        if self.searcher is None:
            self.searcher = self.index.searcher()
        
        query_parser = MultifieldParser(['title', 'content'], self.index.schema)
        parsed_query = query_parser.parse(query)
        
        results = self.searcher.search(parsed_query, limit=limit, scored=True)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for hit in results:
                url = hit['url']
                doc = next((d for d in self.documents if d['url'] == url), None)
                content = doc['content'] if doc else ''
                
                include = True
                if 'min_length' in filters:
                    if len(content) < filters['min_length']:
                        include = False
                if 'must_contain' in filters:
                    if filters['must_contain'].lower() not in content.lower():
                        include = False
                if include:
                    filtered_results.append(hit)
            results = filtered_results
        
        return self._format_results(results)
    
    def visualize_pagerank_scores(self, top_n: int = 20) -> go.Figure:
        """Visualize top PageRank scores."""
        sorted_pr = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        urls, scores = zip(*sorted_pr)
        
        # Shorten URLs for display
        short_urls = [url[:50] + '...' if len(url) > 50 else url for url in urls]
        
        fig = px.bar(
            x=list(short_urls),
            y=list(scores),
            title=f'Top {top_n} Pages by PageRank Score',
            labels={'x': 'URL', 'y': 'PageRank Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def visualize_hits_scores(self, top_n: int = 20) -> go.Figure:
        """Visualize top HITS authority scores."""
        authorities = self.hits_scores.get('authorities', {})
        sorted_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        urls, scores = zip(*sorted_auth)
        
        short_urls = [url[:50] + '...' if len(url) > 50 else url for url in urls]
        
        fig = px.bar(
            x=list(short_urls),
            y=list(scores),
            title=f'Top {top_n} Pages by HITS Authority Score',
            labels={'x': 'URL', 'y': 'Authority Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def visualize_graph(self, G: nx.DiGraph, top_n: int = 30) -> go.Figure:
        """Visualize the URL graph structure."""
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        node_list = [node for node, _ in top_nodes]
        
        subgraph = G.subgraph(node_list)
        
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[:30] + '...' if len(node) > 30 else node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=[degrees[node] for node in subgraph.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Node Degree')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'URL Graph Structure (Top {top_n} Nodes)',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40)
                       ))
        return fig
    
    def compare_ranking_methods(self, query: str, limit: int = 10) -> Dict:
        """Compare different ranking methods for a query."""
        tfidf_results = self.search(query, limit=limit, ranking_method='tfidf')
        pagerank_results = self.search(query, limit=limit, ranking_method='pagerank')
        hits_results = self.search(query, limit=limit, ranking_method='hits')
        
        return {
            'tfidf': tfidf_results,
            'pagerank': pagerank_results,
            'hits': hits_results
        }
    
    def get_document_similarity(self, doc_id1: int, doc_id2: int) -> float:
        """Calculate cosine similarity between two documents."""
        if self.tfidf_matrix is None:
            return 0.0
        
        vec1 = self.tfidf_matrix[doc_id1].toarray()
        vec2 = self.tfidf_matrix[doc_id2].toarray()
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
    
    def close(self):
        """Close the searcher."""
        if self.searcher:
            self.searcher.close()
