import numpy as np
import pandas as pd
from typing import List, Dict, Set
import plotly.graph_objects as go
import plotly.express as px


class IREvaluation:
    """Information Retrieval Evaluation Metrics."""
    
    def __init__(self):
        self.relevant_docs = {}
        self.retrieved_docs = {}
        self.results = {}
        
    def set_relevant_documents(self, query_id: str, relevant_doc_ids: Set[str]):
        """
        Set relevant documents for a query.
        
        Args:
            query_id: Query identifier
            relevant_doc_ids: Set of relevant document IDs
        """
        self.relevant_docs[query_id] = relevant_doc_ids if isinstance(relevant_doc_ids, set) else set(relevant_doc_ids)
    
    def set_retrieved_documents(self, query_id: str, retrieved_doc_ids: List[str]):
        """
        Set retrieved documents for a query.
        
        Args:
            query_id: Query identifier
            retrieved_doc_ids: List of retrieved document IDs in ranked order
        """
        self.retrieved_docs[query_id] = retrieved_doc_ids
    
    def precision_at_k(self, query_id: str, k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            query_id: Query identifier
            k: Cutoff rank
        
        Returns:
            Precision@K score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id][:k]
        relevant = self.relevant_docs[query_id]
        
        relevant_retrieved = len(set(retrieved) & relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(self, query_id: str, k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            query_id: Query identifier
            k: Cutoff rank
        
        Returns:
            Recall@K score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id][:k]
        relevant = self.relevant_docs[query_id]
        
        if len(relevant) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & relevant)
        return relevant_retrieved / len(relevant)
    
    def precision(self, query_id: str) -> float:
        """
        Calculate overall precision.
        
        Args:
            query_id: Query identifier
        
        Returns:
            Precision score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        
        if len(retrieved) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & relevant)
        return relevant_retrieved / len(retrieved)
    
    def recall(self, query_id: str) -> float:
        """
        Calculate overall recall.
        
        Args:
            query_id: Query identifier
        
        Returns:
            Recall score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        
        if len(relevant) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & relevant)
        return relevant_retrieved / len(relevant)
    
    def f1_score(self, query_id: str) -> float:
        """
        Calculate F1-score.
        
        Args:
            query_id: Query identifier
        
        Returns:
            F1-score
        """
        prec = self.precision(query_id)
        rec = self.recall(query_id)
        
        if prec + rec == 0:
            return 0.0
        
        return 2 * (prec * rec) / (prec + rec)
    
    def average_precision(self, query_id: str) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            query_id: Query identifier
        
        Returns:
            Average Precision score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        
        if len(relevant) == 0:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant)
    
    def mean_average_precision(self, query_ids: List[str] = None) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            query_ids: List of query IDs (if None, use all queries)
        
        Returns:
            MAP score
        """
        if query_ids is None:
            query_ids = list(self.relevant_docs.keys())
        
        aps = [self.average_precision(qid) for qid in query_ids]
        return sum(aps) / len(aps) if aps else 0.0
    
    def reciprocal_rank(self, query_id: str) -> float:
        """
        Calculate Reciprocal Rank (RR).
        
        Args:
            query_id: Query identifier
        
        Returns:
            Reciprocal Rank score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def mean_reciprocal_rank(self, query_ids: List[str] = None) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            query_ids: List of query IDs (if None, use all queries)
        
        Returns:
            MRR score
        """
        if query_ids is None:
            query_ids = list(self.relevant_docs.keys())
        
        rrs = [self.reciprocal_rank(qid) for qid in query_ids]
        return sum(rrs) / len(rrs) if rrs else 0.0
    
    def dcg_at_k(self, query_id: str, k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@K (DCG@K).
        
        Args:
            query_id: Query identifier
            k: Cutoff rank
        
        Returns:
            DCG@K score
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return 0.0
        
        retrieved = self.retrieved_docs[query_id][:k]
        relevant = self.relevant_docs[query_id]
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            relevance = 1 if doc_id in relevant else 0
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        return dcg
    
    def ndcg_at_k(self, query_id: str, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
        
        Args:
            query_id: Query identifier
            k: Cutoff rank
        
        Returns:
            NDCG@K score
        """
        dcg = self.dcg_at_k(query_id, k)
        
        # Calculate ideal DCG
        relevant = self.relevant_docs[query_id]
        ideal_retrieved = list(relevant)[:k]
        
        idcg = 0.0
        for i in range(min(k, len(ideal_retrieved))):
            idcg += 1 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_ndcg_at_k(self, k: int, query_ids: List[str] = None) -> float:
        """
        Calculate Mean NDCG@K across queries.
        
        Args:
            k: Cutoff rank
            query_ids: List of query IDs (if None, use all queries)
        
        Returns:
            Mean NDCG@K score
        """
        if query_ids is None:
            query_ids = list(self.relevant_docs.keys())
        
        ndcgs = [self.ndcg_at_k(qid, k) for qid in query_ids]
        return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    
    def calculate_all_metrics(self, query_id: str, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Calculate all metrics for a query.
        
        Args:
            query_id: Query identifier
            k_values: List of k values for precision@k and recall@k
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'precision': self.precision(query_id),
            'recall': self.recall(query_id),
            'f1_score': self.f1_score(query_id),
            'average_precision': self.average_precision(query_id),
            'reciprocal_rank': self.reciprocal_rank(query_id)
        }
        
        for k in k_values:
            metrics[f'precision_at_{k}'] = self.precision_at_k(query_id, k)
            metrics[f'recall_at_{k}'] = self.recall_at_k(query_id, k)
            metrics[f'ndcg_at_{k}'] = self.ndcg_at_k(query_id, k)
        
        return metrics
    
    def calculate_system_metrics(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Calculate system-wide metrics across all queries.
        
        Args:
            k_values: List of k values for precision@k and recall@k
        
        Returns:
            Dictionary with system-wide metrics
        """
        query_ids = list(self.relevant_docs.keys())
        
        metrics = {
            'map': self.mean_average_precision(query_ids),
            'mrr': self.mean_reciprocal_rank(query_ids)
        }
        
        for k in k_values:
            metrics[f'mean_precision_at_{k}'] = np.mean([self.precision_at_k(qid, k) for qid in query_ids])
            metrics[f'mean_recall_at_{k}'] = np.mean([self.recall_at_k(qid, k) for qid in query_ids])
            metrics[f'mean_ndcg_at_{k}'] = self.mean_ndcg_at_k(k, query_ids)
        
        return metrics
    
    def get_per_query_metrics(self, k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Get metrics for each query as a DataFrame.
        
        Args:
            k_values: List of k values
        
        Returns:
            DataFrame with per-query metrics
        """
        query_ids = list(self.relevant_docs.keys())
        all_metrics = []
        
        for qid in query_ids:
            metrics = self.calculate_all_metrics(qid, k_values)
            metrics['query_id'] = qid
            all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)
    
    def visualize_precision_recall_curve(self, query_id: str) -> go.Figure:
        """
        Visualize precision-recall curve for a query.
        
        Args:
            query_id: Query identifier
        
        Returns:
            Plotly figure
        """
        if query_id not in self.retrieved_docs or query_id not in self.relevant_docs:
            return go.Figure()
        
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        
        precisions = []
        recalls = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_count += 1
            
            precision = relevant_count / (i + 1)
            recall = relevant_count / len(relevant) if len(relevant) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recalls,
            y=precisions,
            mode='lines',
            name='Precision-Recall Curve'
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve for Query: {query_id}',
            xaxis_title='Recall',
            yaxis_title='Precision'
        )
        
        return fig
    
    def visualize_metrics_comparison(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Visualize metrics comparison across queries.
        
        Args:
            metrics_df: DataFrame with per-query metrics
        
        Returns:
            Plotly figure
        """
        # Select key metrics for comparison
        key_metrics = ['precision', 'recall', 'f1_score', 'average_precision', 'reciprocal_rank']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        fig = go.Figure()
        
        for metric in available_metrics:
            fig.add_trace(go.Box(
                y=metrics_df[metric],
                name=metric.replace('_', ' ').title()
            ))
        
        fig.update_layout(
            title='Metrics Distribution Across Queries',
            yaxis_title='Score',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def visualize_ndcg_at_k(self, k_values: List[int] = [5, 10, 20]) -> go.Figure:
        """
        Visualize NDCG@K across queries.
        
        Args:
            k_values: List of k values
        
        Returns:
            Plotly figure
        """
        query_ids = list(self.relevant_docs.keys())
        
        fig = go.Figure()
        
        for k in k_values:
            ndcg_values = [self.ndcg_at_k(qid, k) for qid in query_ids]
            fig.add_trace(go.Box(
                y=ndcg_values,
                name=f'NDCG@{k}'
            ))
        
        fig.update_layout(
            title='NDCG@K Distribution Across Queries',
            yaxis_title='NDCG Score',
            xaxis_title='K Value'
        )
        
        return fig
    
    def compare_ranking_methods(self, method_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare different ranking methods.
        
        Args:
            method_results: Dictionary mapping method names to their retrieved results
        
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for method_name, retrieved_dict in method_results.items():
            # Temporarily set retrieved docs
            original_retrieved = self.retrieved_docs.copy()
            self.retrieved_docs = retrieved_dict
            
            metrics = self.calculate_system_metrics()
            metrics['method'] = method_name
            comparison.append(metrics)
            
            # Restore original
            self.retrieved_docs = original_retrieved
        
        return pd.DataFrame(comparison)
    
    def visualize_method_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Visualize comparison of ranking methods.
        
        Args:
            comparison_df: DataFrame from compare_ranking_methods
        
        Returns:
            Plotly figure
        """
        methods = comparison_df['method'].tolist()
        
        # Select key metrics
        key_metrics = ['map', 'mrr', 'mean_precision_at_10', 'mean_ndcg_at_10']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        fig = go.Figure()
        
        for metric in available_metrics:
            fig.add_trace(go.Bar(
                x=methods,
                y=comparison_df[metric],
                name=metric.replace('_', ' ').title()
            ))
        
        fig.update_layout(
            title='Ranking Methods Comparison',
            barmode='group',
            xaxis_tickangle=-45,
            yaxis_title='Score'
        )
        
        return fig
