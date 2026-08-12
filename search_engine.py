import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px


class SearchEngine:
    def __init__(self):
        self.documents = []
        self.url_graph = {}
        self.pagerank_scores = {}
        self.hits_scores = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_docs = []
        
    def create_index(self, documents: List[Dict], metadata: List[Dict] = None, processed_docs: List[str] = None):
        """Create TF-IDF index from documents."""
        self.documents = documents
        
        # Use processed docs if provided, otherwise use raw content
        if processed_docs is None:
            from text_preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()
            self.processed_docs = [preprocessor.preprocess_pipeline(doc['content']) for doc in documents]
        else:
            self.processed_docs = processed_docs
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_docs)
        
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
        if self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        from text_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        processed_query = preprocessor.preprocess_pipeline(query)
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = similarities.argsort()[::-1][:limit * 2]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': idx,
                'score': similarities[idx]
            })
        
        if ranking_method == 'pagerank':
            return self._rerank_with_pagerank(results, limit)
        elif ranking_method == 'hits':
            return self._rerank_with_hits(results, limit)
        else:
            return self._format_results(results[:limit])
    
    def _rerank_with_pagerank(self, results, limit: int) -> List[Dict]:
        """Re-rank results using PageRank scores."""
        reranked = []
        for hit in results:
            doc_id = hit['doc_id']
            if doc_id >= len(self.documents):
                continue
            
            doc = self.documents[doc_id]
            url = doc['url']
            base_score = hit['score']
            pr_score = self.pagerank_scores.get(url, 0)
            combined_score = 0.7 * base_score + 0.3 * pr_score
            
            content = doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content']
            
            reranked.append({
                'url': url,
                'title': '',
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
            doc_id = hit['doc_id']
            if doc_id >= len(self.documents):
                continue
            
            doc = self.documents[doc_id]
            url = doc['url']
            base_score = hit['score']
            auth_score = authorities.get(url, 0)
            combined_score = 0.7 * base_score + 0.3 * auth_score
            
            content = doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content']
            
            reranked.append({
                'url': url,
                'title': '',
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
            doc_id = hit['doc_id']
            if doc_id >= len(self.documents):
                continue
            
            doc = self.documents[doc_id]
            content = doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content']
            
            formatted.append({
                'url': doc['url'],
                'title': '',
                'score': hit['score'],
                'content': content
            })
        return formatted
    
    def advanced_search(self, query: str, filters: Dict = None, limit: int = 10) -> List[Dict]:
        """Advanced search with filters."""
        results = self.search(query, limit=limit * 2)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                doc = next((d for d in self.documents if d['url'] == result['url']), None)
                if not doc:
                    continue
                
                content = doc['content']
                
                include = True
                if 'min_length' in filters:
                    if len(content) < filters['min_length']:
                        include = False
                if 'must_contain' in filters:
                    if filters['must_contain'].lower() not in content.lower():
                        include = False
                if include:
                    filtered_results.append(result)
            results = filtered_results[:limit]
        else:
            results = results[:limit]
        
        return results
    
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
        """No-op for TF-IDF based search."""
        pass
