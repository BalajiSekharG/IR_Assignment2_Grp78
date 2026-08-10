import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import re
import time
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from text_preprocessing import TextPreprocessor, safe_df_bounds


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range, handling all-zero vectors."""
    max_val = scores.max()
    if max_val > 0:
        return scores / max_val
    return scores

class SearchEngine:
    def __init__(self):
        self.documents = []
        self.url_graph = {}
        self.pagerank_scores = {}
        self.hits_scores = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_docs = []
        self.preprocessor = TextPreprocessor()
        self.metadata_by_url = {}
        self.index_stats = {}
        self.last_query_info = {}
        
    def create_index(self, documents: List[Dict], metadata: List[Dict] = None, processed_docs: List[str] = None):
        """Create TF-IDF index from documents."""
        self.documents = documents
        
        # Use processed docs if provided, otherwise use raw content
        if processed_docs is None:
            self.processed_docs = [self.preprocessor.preprocess_pipeline(doc['content']) for doc in documents]
        else:
            self.processed_docs = processed_docs
        
        # Store metadata by URL for filtering
        if metadata:
            self.metadata_by_url = {meta.get('url', ''): meta for meta in metadata}
        else:
            self.metadata_by_url = {doc['url']: {} for doc in documents}
        
        # Create TF-IDF vectorizer with adaptive bounds
        min_df, max_df = safe_df_bounds(len(self.processed_docs))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=min_df, max_df=max_df)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_docs)
        
        # Build BM25 components
        self._build_bm25()
        
        # Store index statistics
        self.index_stats = {
            'num_docs': len(documents),
            'vocab_size': len(self.tfidf_vectorizer.get_feature_names_out()),
            'avg_doc_length': np.mean([len(doc.split()) for doc in self.processed_docs]),
            'total_terms': sum(len(doc.split()) for doc in self.processed_docs)
        }
        
    def _build_bm25(self):
        """Build BM25 ranking components."""
        self.count_vectorizer = CountVectorizer()
        self.term_frequencies = self.count_vectorizer.fit_transform(self.processed_docs)
        self.doc_lengths = np.asarray(self.term_frequencies.sum(axis=1)).ravel()
        self.avg_doc_length = np.mean(self.doc_lengths)
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        
        # Precompute IDF for BM25
        df = np.asarray((self.term_frequencies > 0).sum(axis=0)).ravel()
        self.bm25_idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1.0)
    
    def _bm25_scores(self, query_terms: List[str]) -> np.ndarray:
        """Calculate BM25 scores for query terms."""
        if not hasattr(self, 'term_frequencies'):
            return np.zeros(len(self.documents))
        
        vocab = self.count_vectorizer.get_feature_names_out()
        term_to_idx = {term: idx for idx, term in enumerate(vocab)}
        
        scores = np.zeros(len(self.documents))
        for term in query_terms:
            if term in term_to_idx:
                term_idx = term_to_idx[term]
                tf = self.term_frequencies[:, term_idx].toarray().ravel()
                idf = self.bm25_idf[term_idx]
                
                numerator = tf * (self.bm25_k1 + 1)
                denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (self.doc_lengths / self.avg_doc_length))
                scores += idf * (numerator / denominator)
        
        return scores
    
    def parse_query(self, query: str) -> Dict:
        """Parse query with operators (phrases, +required, -excluded)."""
        phrases = re.findall(r'"([^"]+)"', query)
        required = re.findall(r'\+([^\s]+)', query)
        excluded = re.findall(r'-([^\s]+)', query)
        
        # Remove operators from query for basic term extraction
        clean_query = re.sub(r'["\+\-]', ' ', query)
        terms = clean_query.split()
        
        return {
            'terms': terms,
            'phrases': phrases,
            'required': required,
            'excluded': excluded
        }
    
    def _apply_query_operators(self, results: List[Dict], parsed_query: Dict) -> List[Dict]:
        """Filter results based on query operators."""
        filtered = []
        for result in results:
            doc = self.documents[result['doc_id']]
            content = doc['content'].lower()
            
            # Check required terms
            for term in parsed_query['required']:
                if term.lower() not in content:
                    break
            else:
                # Check excluded terms
                for term in parsed_query['excluded']:
                    if term.lower() in content:
                        break
                else:
                    filtered.append(result)
        
        return filtered
    
    def expand_query(self, query: str, top_k: int = 5) -> str:
        """Expand query using pseudo-relevance feedback."""
        initial_results = self.search(query, limit=top_k, ranking_method='tfidf')
        if not initial_results:
            return query
        
        # Extract terms from top results
        top_doc_indices = [r['doc_id'] for r in initial_results]
        top_docs = [self.processed_docs[i] for i in top_doc_indices]
        
        # Get top terms from these documents
        all_terms = ' '.join(top_docs).split()
        term_freq = Counter(all_terms)
        
        # Get top frequent terms not already in query
        query_terms = set(query.lower().split())
        expansion_terms = [term for term, _ in term_freq.most_common(top_k) if term not in query_terms]
        
        if expansion_terms:
            expanded = query + ' ' + ' '.join(expansion_terms)
            return expanded
        
        return query
    
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
        if G.number_of_nodes() == 0:
            return {}, {}
        hubs, authorities = nx.hits(G, max_iter=max_iter)
        self.hits_scores = {'hubs': hubs, 'authorities': authorities}
        return hubs, authorities
    
    def _link_scores(self, doc_ids: List[int]) -> Dict[str, np.ndarray]:
        """Get link-analysis scores for documents."""
        pr_scores = np.zeros(len(doc_ids))
        auth_scores = np.zeros(len(doc_ids))
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id < len(self.documents):
                url = self.documents[doc_id]['url']
                pr_scores[i] = self.pagerank_scores.get(url, 0)
                auth_scores[i] = self.hits_scores.get('authorities', {}).get(url, 0)
        
        return {'pagerank': pr_scores, 'authority': auth_scores}
    
    def search(self, query: str, limit: int = 10, ranking_method: str = 'tfidf', 
                expand: bool = False, filters: Dict = None, link_weight: float = 0.3) -> List[Dict]:
        """
        Search documents with specified ranking method.
        
        Args:
            query: Search query
            limit: Number of results to return
            ranking_method: 'tfidf', 'bm25', 'pagerank', or 'hits'
            expand: Whether to use query expansion
            filters: Post-search filters (min_length, max_length, must_contain, must_not_contain, category, min_score)
            link_weight: Weight for link-analysis scores in hybrid ranking
        """
        start_time = time.time()
        
        if self.tfidf_matrix is None:
            return []
        
        # Parse query for operators
        parsed = self.parse_query(query)
        
        # Apply query expansion if requested
        if expand:
            query = self.expand_query(query)
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_pipeline(query)
        query_terms = processed_query.split()
        
        # Calculate text scores based on ranking method
        if ranking_method == 'bm25':
            text_scores = self._bm25_scores(query_terms)
        else:
            # TF-IDF cosine similarity
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            text_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top candidates
        top_indices = text_scores.argsort()[::-1][:limit * 3]
        
        # Build initial results
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': idx,
                'score': text_scores[idx]
            })
        
        # Apply query operators
        results = self._apply_query_operators(results, parsed)
        
        # Apply hybrid ranking with link analysis if available
        if ranking_method in ['pagerank', 'hits'] and self.pagerank_scores:
            doc_ids = [r['doc_id'] for r in results]
            link_scores = self._link_scores(doc_ids)
            
            for i, result in enumerate(results):
                text_score = result['score']
                if ranking_method == 'pagerank':
                    link_score = link_scores['pagerank'][i]
                else:
                    link_score = link_scores['authority'][i]
                
                # Normalize and fuse
                text_norm = normalize_scores(np.array([text_score]))[0]
                link_norm = normalize_scores(np.array([link_score]))[0]
                result['score'] = (1 - link_weight) * text_norm + link_weight * link_norm
                result['original_score'] = text_score
                result['link_score'] = link_score
        
        # Sort by final score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply post-search filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Format results with metadata
        formatted = []
        for rank, result in enumerate(results[:limit]):
            doc_id = result['doc_id']
            if doc_id >= len(self.documents):
                continue
            
            doc = self.documents[doc_id]
            content = doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content']
            
            formatted_result = {
                'doc_id': doc_id,
                'rank': rank + 1,
                'url': doc['url'],
                'title': doc.get('title', ''),
                'score': round(result['score'], 4),
                'content': content
            }
            
            if 'original_score' in result:
                formatted_result['original_score'] = round(result['original_score'], 4)
            if 'link_score' in result:
                formatted_result['link_score'] = round(result['link_score'], 4)
            
            formatted.append(formatted_result)
        
        # Store query info for analysis
        self.last_query_info = {
            'query': query,
            'ranking_method': ranking_method,
            'num_results': len(formatted),
            'query_time': time.time() - start_time,
            'expanded': expand,
            'filters_applied': filters is not None
        }
        
        return formatted
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Filter a result list on content and metadata constraints."""
        filtered = []
        for result in results:
            doc = self.documents[result['doc_id']]
            content = doc['content']
            
            if 'min_length' in filters and len(content.split()) < filters['min_length']:
                continue
            if 'max_length' in filters and len(content.split()) > filters['max_length']:
                continue
            if 'must_contain' in filters and filters['must_contain'].lower() not in content.lower():
                continue
            if 'must_not_contain' in filters and filters['must_not_contain'].lower() in content.lower():
                continue
            if 'category' in filters and filters['category'] and result.get('category') != filters['category']:
                continue
            if 'min_score' in filters and result['score'] < filters['min_score']:
                continue
            
            filtered.append(result)
        
        return filtered
    
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
        if not self.pagerank_scores:
            return go.Figure(layout=go.Layout(
                title='No PageRank scores available - build the graph first'))
        
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
        """Visualize top HITS hub and authority scores side by side."""
        authorities = self.hits_scores.get('authorities', {})
        hubs = self.hits_scores.get('hubs', {})
        
        if not authorities:
            return go.Figure(layout=go.Layout(
                title='No HITS scores available - build the graph first'))
        
        sorted_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        urls, scores = zip(*sorted_auth)
        
        short_urls = [url[:50] + '...' if len(url) > 50 else url for url in urls]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(short_urls), y=list(scores), name='Authority'))
        fig.add_trace(go.Bar(x=list(short_urls),
                             y=[hubs.get(url, 0) for url in urls], name='Hub'))
        fig.update_layout(
            title=f'Top {top_n} Pages by HITS Score',
            barmode='group',
            xaxis_tickangle=-45,
            yaxis_title='Score'
        )
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
    
    def compare_ranking_methods(self, query: str, limit: int = 10,
                                methods: List[str] = None,
                                link_weight: float = 0.3) -> Dict[str, List[Dict]]:
        """Run the same query under several ranking methods."""
        if methods is None:
            methods = ['tfidf', 'bm25', 'pagerank', 'hits']
        
        return {
            method: self.search(query, limit=limit, ranking_method=method,
                               link_weight=link_weight)
            for method in methods
        }
    
    def ranking_comparison_table(self, query: str, limit: int = 10,
                                  methods: List[str] = None,
                                  link_weight: float = 0.3) -> pd.DataFrame:
        """Build a rank-by-method table showing how the ordering changes.

        This is the evidence that ranking matters: the same candidate set
        is retrieved, but each method places the documents at different ranks.
        """
        comparison = self.compare_ranking_methods(query, limit=limit, methods=methods,
                                                 link_weight=link_weight)
        
        rows = {}
        for method, results in comparison.items():
            for result in results:
                row = rows.setdefault(result['doc_id'], {
                    'doc_id': result['doc_id'],
                    'title': result['title'] or result['url'],
                    'url': result['url']
                })
                row[f'{method}_rank'] = result['rank']
                row[f'{method}_score'] = round(result['score'], 4)
        
        table = pd.DataFrame(list(rows.values()))
        if table.empty:
            return table
        
        rank_columns = [c for c in table.columns if c.endswith('_rank')]
        table[rank_columns] = table[rank_columns].fillna(0).astype(int)
        # 0 means "not retrieved by this method", so it must sort *last"
        # rather than ahead of rank 1. Sort on the best rank any method assigned.
        best_rank = table[rank_columns].replace(0, np.nan).min(axis=1)
        table = table.assign(_best=best_rank).sort_values(
            '_best', ascending=True, na_position='last').drop(columns='_best').reset_index(drop=True)
        
        if len(rank_columns) >= 2:
            table['max_rank_shift'] = table[rank_columns].replace(0, np.nan).max(axis=1) - \
                                      table[rank_columns].replace(0, np.nan).min(axis=1)
        
        return table
    
    def visualize_ranking_comparison(self, comparison_table: pd.DataFrame) -> go.Figure:
        """Visualize the rank each method assigns to each document."""
        if comparison_table.empty:
            return go.Figure(layout=go.Layout(title='No results to compare'))
        
        rank_columns = [c for c in comparison_table.columns if c.endswith('_rank')]
        labels = [
            (title[:40] + '...') if len(str(title)) > 40 else str(title)
            for title in comparison_table['title']
        ]
        
        fig = go.Figure()
        for column in rank_columns:
            fig.add_trace(go.Bar(
                x=labels,
                y=comparison_table[column],
                name=column.replace('_rank', '').upper()
            ))
        
        fig.update_layout(
            title='Rank Assigned to the Same Document by Each Ranking Method',
            barmode='group',
            xaxis_tickangle=-30,
            yaxis_title='Rank (1 = best, 0 = not retrieved)',
            yaxis=dict(autorange='reversed')
        )
        return fig
    
    def get_index_statistics(self) -> Dict:
        """Return statistics describing the current index."""
        return dict(self.index_stats)
    
    def get_top_terms(self, top_n: int = 20) -> pd.DataFrame:
        """Return the highest-weight terms in the index (index inspection)."""
        if self.tfidf_matrix is None:
            return pd.DataFrame()
        
        vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        weights = np.asarray(self.tfidf_matrix.sum(axis=0)).ravel()
        doc_freq = np.asarray((self.tfidf_matrix > 0).sum(axis=0)).ravel()
        order = weights.argsort()[::-1][:top_n]
        
        return pd.DataFrame({
            'term': [vocabulary[i] for i in order],
            'total_tfidf_weight': [round(float(weights[i]), 4) for i in order],
            'document_frequency': [int(doc_freq[i]) for i in order]
        })
    
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
