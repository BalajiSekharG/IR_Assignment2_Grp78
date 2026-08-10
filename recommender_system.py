import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px

from text_preprocessing import TextPreprocessor, safe_df_bounds


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise a score vector onto [0, 1] for fair fusion."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    lowest, highest = float(scores.min()), float(scores.max())
    if highest - lowest < 1e-12:
        return np.zeros_like(scores)
    return (scores - lowest) / (highest - lowest)


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.vectorizer = None
        self.documents = []
        self.document_urls = []
        self.processed_docs = []
        self.similarity_matrix = None
        
    def fit(self, documents: List[Dict], processed_docs: List[str] = None):
        """
        Fit the recommender with documents.
        
        Args:
            documents: List of document dictionaries with 'url' and 'content'
            processed_docs: Optional preprocessed documents
        """
        self.documents = documents
        self.document_urls = [doc['url'] for doc in documents]
        
        if processed_docs is None or len(processed_docs) != len(documents):
            preprocessor = TextPreprocessor()
            processed_docs = [preprocessor.preprocess_pipeline(doc['content']) for doc in documents]
        self.processed_docs = processed_docs
        
        # Create TF-IDF matrix with bounds that are safe for small corpora
        min_df, max_df = safe_df_bounds(len(processed_docs))
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=min_df, max_df=max_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
    
    def content_scores(self, doc_index: int) -> np.ndarray:
        """Return content similarity of every document to the given document."""
        if self.similarity_matrix is None or doc_index >= len(self.documents):
            return np.zeros(len(self.documents))
        return np.asarray(self.similarity_matrix[doc_index], dtype=float)
    
    def recommend(self, doc_index: int, top_k: int = 5) -> List[Dict]:
        """
        Recommend similar documents based on a given document.
        
        Args:
            doc_index: Index of the query document
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendations with similarity scores
        """
        if doc_index >= len(self.documents):
            return []
        
        # Get similarity scores for the query document
        sim_scores = self.similarity_matrix[doc_index]
        
        # Rank all other documents, always excluding the query document itself
        candidates = [i for i in sim_scores.argsort()[::-1] if i != doc_index]
        top_indices = candidates[:top_k]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'doc_id': int(idx),
                'url': self.document_urls[idx],
                'title': self.documents[idx].get('title', ''),
                'similarity_score': float(sim_scores[idx]),
                'approach': 'content',
                'content_preview': self.documents[idx]['content'][:200] + '...' if len(self.documents[idx]['content']) > 200 else self.documents[idx]['content']
            })
        
        return recommendations
    
    def recommend_by_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recommend documents based on a text query.
        
        Args:
            query: Text query
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendations with similarity scores
        """
        if self.vectorizer is None:
            return []
        
        # The query must go through the same preprocessing as the documents,
        # otherwise raw query tokens will not match the indexed vocabulary.
        preprocessor = TextPreprocessor()
        processed_query = preprocessor.preprocess_pipeline(query)
        if not processed_query.strip():
            return []
        
        query_vector = self.vectorizer.transform([processed_query])
        
        # Compute similarity with all documents
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get indices of top-k similar documents
        top_indices = [i for i in sim_scores.argsort()[::-1] if sim_scores[i] > 0][:top_k]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'doc_id': int(idx),
                'url': self.document_urls[idx],
                'title': self.documents[idx].get('title', ''),
                'similarity_score': float(sim_scores[idx]),
                'approach': 'content',
                'content_preview': self.documents[idx]['content'][:200] + '...' if len(self.documents[idx]['content']) > 200 else self.documents[idx]['content']
            })
        
        return recommendations
    
    def get_document_similarity(self, doc_index1: int, doc_index2: int) -> float:
        """Get similarity between two documents."""
        return self.similarity_matrix[doc_index1][doc_index2]
    
    def visualize_recommendations(self, recommendations: List[Dict]) -> go.Figure:
        """Visualize recommendations with similarity scores."""
        urls = [rec['url'][:50] + '...' if len(rec['url']) > 50 else rec['url'] for rec in recommendations]
        scores = [rec['similarity_score'] for rec in recommendations]
        
        fig = px.bar(
            x=list(urls),
            y=list(scores),
            title='Document Recommendations with Similarity Scores',
            labels={'x': 'Document URL', 'y': 'Similarity Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig


class CollaborativeRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.users = []
        self.items = []
        self.item_positions = {}
        self.ratings_data = []
        
    def fit(self, ratings_data: List[Dict]):
        """
        Fit collaborative filtering model.
        
        Args:
            ratings_data: List of dictionaries with 'user_id', 'item_id', 'rating'
        """
        if not ratings_data:
            return
        
        self.ratings_data = ratings_data
        df = pd.DataFrame(ratings_data)
        
        # Create user-item matrix
        self.user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.users = self.user_item_matrix.index.tolist()
        self.items = self.user_item_matrix.columns.tolist()
        self.item_positions = {item: i for i, item in enumerate(self.items)}
        
        # Compute user-user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
    
    def is_fitted(self) -> bool:
        """Whether any ratings were supplied."""
        return self.user_item_matrix is not None and len(self.items) > 0
    
    def item_scores_for_documents(self, item_id: str, document_urls: List[str]) -> np.ndarray:
        """Return the item-item collaborative score of every document.

        The score of document *d* is the co-rating similarity between *d* and
        the query item, i.e. "users who liked this also liked that". Documents
        that nobody rated receive a score of zero, which is exactly the cold
        start weakness of collaborative filtering.
        """
        scores = np.zeros(len(document_urls), dtype=float)
        if not self.is_fitted():
            return scores
        
        query_position = self.item_positions.get(item_id)
        if query_position is None:
            return scores
        
        for i, url in enumerate(document_urls):
            position = self.item_positions.get(url)
            if position is not None and position != query_position:
                scores[i] = float(self.item_similarity[query_position, position])
        return scores
    
    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> List[Dict]:
        """Recommend items co-rated with the given item (item-based CF)."""
        if not self.is_fitted():
            return []
        
        query_position = self.item_positions.get(item_id)
        if query_position is None:
            return []
        
        similarities = self.item_similarity[query_position]
        order = [i for i in similarities.argsort()[::-1]
                  if i != query_position and similarities[i] > 0][:top_k]
        
        return [
            {
                'item_id': self.items[i],
                'url': self.items[i],
                'similarity_score': float(similarities[i]),
                'approach': 'collaborative'
            }
            for i in order
        ]
    
    def get_coverage(self, document_urls: List[str]) -> Dict:
        """Report how much of the collection the ratings actually cover."""
        if not self.is_fitted():
            return {'rated_documents': 0, 'total_documents': len(document_urls),
                    'coverage_%': 0.0, 'users': 0, 'ratings': 0,
                    'matrix_sparsity_%': 100.0}
        
        rated = sum(1 for url in document_urls if url in self.item_positions)
        cells = self.user_item_matrix.size
        non_zero = int((self.user_item_matrix.values > 0).sum())
        return {
            'rated_documents': rated,
            'total_documents': len(document_urls),
            'coverage_%': round(100 * rated / len(document_urls), 1) if document_urls else 0.0,
            'users': len(self.users),
            'ratings': non_zero,
            'matrix_sparsity_%': round(100 * (1 - non_zero / cells), 1) if cells else 100.0
        }
    
    def visualize_user_item_matrix(self) -> go.Figure:
        """Visualize the user-item rating matrix."""
        if not self.is_fitted():
            return go.Figure(layout=go.Layout(title='No ratings data available'))
        
        short_items = [item.split('/')[-1][:28] for item in self.items]
        fig = px.imshow(
            self.user_item_matrix.values,
            x=short_items,
            y=self.users,
            color_continuous_scale='Blues',
            labels={'x': 'Item', 'y': 'User', 'color': 'Rating'},
            title='User-Item Rating Matrix (0 = not rated)'
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def recommend_user_based(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """
        Recommend items using user-based collaborative filtering.
        
        Args:
            user_id: User to recommend for
            top_k: Number of recommendations
        
        Returns:
            List of recommended items with predicted ratings
        """
        if user_id not in self.users:
            return []
        
        user_idx = self.users.index(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        # Get items the user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_items = user_ratings[user_ratings == 0].index
        
        recommendations = []
        for item in unrated_items:
            item_idx = self.items.index(item)
            
            # Predict rating using weighted average of similar users
            weighted_sum = 0
            similarity_sum = 0
            
            for other_user_idx, other_user in enumerate(self.users):
                if other_user_idx == user_idx:
                    continue
                
                other_rating = self.user_item_matrix.iloc[other_user_idx, item_idx]
                if other_rating > 0:
                    similarity = user_similarities[other_user_idx]
                    weighted_sum += similarity * other_rating
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append({
                    'item_id': item,
                    'url': item,
                    'predicted_rating': predicted_rating,
                    'similarity_score': predicted_rating,
                    'approach': 'collaborative_user_based'
                })
        
        # Sort by predicted rating and return top-k
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:top_k]
    
    def recommend_item_based(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """
        Recommend items using item-based collaborative filtering.
        
        Args:
            user_id: User to recommend for
            top_k: Number of recommendations
        
        Returns:
            List of recommended items with predicted ratings
        """
        if user_id not in self.users:
            return []
        
        user_idx = self.users.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get items the user has rated
        rated_items = user_ratings[user_ratings > 0]
        
        # Get items the user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        recommendations = []
        for item in unrated_items:
            item_idx = self.items.index(item)
            
            # Predict rating using similar items
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item, rating in rated_items.items():
                rated_item_idx = self.items.index(rated_item)
                similarity = self.item_similarity[item_idx, rated_item_idx]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append({
                    'item_id': item,
                    'url': item,
                    'predicted_rating': predicted_rating,
                    'similarity_score': predicted_rating,
                    'approach': 'collaborative_item_based'
                })
        
        # Sort by predicted rating and return top-k
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:top_k]


class HybridRecommender:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.documents = []
        
    def fit(self, documents: List[Dict], processed_docs: List[str] = None, 
            ratings_data: List[Dict] = None):
        """
        Fit hybrid recommender with both content and collaborative data.
        
        Args:
            documents: List of document dictionaries
            processed_docs: Preprocessed documents for content-based
            ratings_data: Ratings data for collaborative filtering
        """
        self.documents = documents
        
        # Fit content-based recommender
        self.content_recommender.fit(documents, processed_docs)
        
        # Fit collaborative recommender if ratings data is provided
        if ratings_data:
            self.collaborative_recommender.fit(ratings_data)
    
    def has_collaborative_signal(self) -> bool:
        """Whether a usable collaborative component was fitted."""
        return self.collaborative_recommender.is_fitted()
    
    def recommend(self, doc_index: int, top_k: int = 5, 
                  content_weight: float = 0.7, collab_weight: float = 0.3) -> List[Dict]:
        """
        Recommend using a genuine hybrid of content and collaborative signals.

        Both components are computed over the whole collection and min-max
        normalised before the weighted sum, so the collaborative signal really
        changes the ordering instead of being a cosmetic zero column.

        Args:
            doc_index: Index of query document
            top_k: Number of recommendations
            content_weight: Weight for content-based scores
            collab_weight: Weight for collaborative scores

        Returns:
            List of hybrid recommendations with per-component contributions
        """
        if not self.documents or doc_index >= len(self.documents):
            return []

        document_urls = [doc['url'] for doc in self.documents]
        query_url = document_urls[doc_index]

        content_scores = self.content_recommender.content_scores(doc_index)
        collab_scores = self.collaborative_recommender.item_scores_for_documents(
            query_url, document_urls)

        # Renormalise the weights when no ratings are available, so the hybrid
        # degrades gracefully to pure content-based instead of shrinking scores.
        if not self.has_collaborative_signal() or collab_scores.max() <= 0:
            content_weight, collab_weight = 1.0, 0.0

        normalized_content = normalize_scores(content_scores)
        normalized_collab = normalize_scores(collab_scores)
        combined = content_weight * normalized_content + collab_weight * normalized_collab

        combined[doc_index] = -1.0  # never recommend the query document
        order = [i for i in combined.argsort()[::-1] if combined[i] > 0][:top_k]

        recommendations = []
        for index in order:
            doc = self.documents[index]
            recommendations.append({
                'doc_id': int(index),
                'url': doc['url'],
                'title': doc.get('title', ''),
                'similarity_score': float(combined[index]),
                'content_score': float(content_scores[index]),
                'collab_score': float(collab_scores[index]),
                'content_contribution': float(content_weight * normalized_content[index]),
                'collab_contribution': float(collab_weight * normalized_collab[index]),
                'approach': 'hybrid',
                'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
            })

        return recommendations
    
    def visualize_hybrid_recommendations(self, recommendations: List[Dict]) -> go.Figure:
        """Visualize hybrid recommendations showing contributions."""
        urls = [rec['url'][:40] + '...' if len(rec['url']) > 40 else rec['url'] for rec in recommendations]
        content_contrib = [rec['content_contribution'] for rec in recommendations]
        collab_contrib = [rec['collab_contribution'] for rec in recommendations]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Content-based',
            x=list(urls),
            y=list(content_contrib),
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.add_trace(go.Bar(
            name='Collaborative',
            x=list(urls),
            y=list(collab_contrib),
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig.update_layout(
            title='Hybrid Recommendations - Contribution Breakdown',
            barmode='stack',
            xaxis_tickangle=-45,
            yaxis_title='Contribution Score'
        )
        
        return fig


class RecommenderSystem:
    """Main recommender system that supports multiple approaches."""
    
    def __init__(self):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeRecommender()
        self.hybrid = HybridRecommender()
        self.approach = 'content'
        self.documents = []
        self.ratings_data = []
        self.fitted_signature = None
        
    def fit(self, documents: List[Dict], processed_docs: List[str] = None,
            ratings_data: List[Dict] = None, approach: str = 'content'):
        """
        Fit the recommender system.

        All three sub-models are fitted whenever the data is available, which
        makes it possible to compare the approaches side by side without
        refitting.

        Args:
            documents: List of document dictionaries
            processed_docs: Preprocessed documents
            ratings_data: Ratings data for collaborative filtering
            approach: 'content', 'collaborative', or 'hybrid'
        """
        self.approach = approach
        self.documents = documents
        self.ratings_data = ratings_data or []

        self.content_based.fit(documents, processed_docs)
        if ratings_data:
            self.collaborative.fit(ratings_data)
        self.hybrid.fit(documents, processed_docs, ratings_data)

        # Signature lets the UI skip an expensive refit on every rerun.
        self.fitted_signature = (len(documents), len(self.ratings_data),
                                documents[0]['url'] if documents else None)

    def needs_fit(self, documents: List[Dict], ratings_data: List[Dict] = None) -> bool:
        """Whether the model must be refitted for this data."""
        signature = (len(documents), len(ratings_data or []),
                    documents[0]['url'] if documents else None)
        return signature != self.fitted_signature
    
    def recommend(self, doc_index: int = None, query: str = None, 
                  user_id: str = None, top_k: int = 5,
                  content_weight: float = 0.7,
                  collab_weight: float = 0.3) -> List[Dict]:
        """
        Generate recommendations for the currently selected approach.

        Args:
            doc_index: Index of document (content, collaborative item-based, hybrid)
            query: Text query (content-based only)
            user_id: User ID (collaborative user-based)
            top_k: Number of recommendations
            content_weight: Content weight for the hybrid approach
            collab_weight: Collaborative weight for the hybrid approach

        Returns:
            List of recommendations
        """
        if self.approach == 'content':
            if query:
                return self.content_based.recommend_by_query(query, top_k)
            if doc_index is not None:
                return self.content_based.recommend(doc_index, top_k)
        elif self.approach == 'collaborative':
            if user_id:
                return self.collaborative.recommend_user_based(user_id, top_k)
            if doc_index is not None and doc_index < len(self.documents):
                return self.collaborative.recommend_similar_items(
                    self.documents[doc_index]['url'], top_k)
        elif self.approach == 'hybrid':
            if doc_index is not None:
                return self.hybrid.recommend(doc_index, top_k,
                                                 content_weight=content_weight,
                                                 collab_weight=collab_weight)

        return []
    
    def compare_approaches(self, doc_index: int, top_k: int = 5) -> pd.DataFrame:
        """Compare content-based, collaborative and hybrid Top-K side by side.

        Directly supports the required discussion of when each approach is
        preferable: content-based always returns results, collaborative only
        covers rated items, and the hybrid inherits the strengths of both.
        """
        if not self.documents or doc_index >= len(self.documents):
            return pd.DataFrame()

        query_url = self.documents[doc_index]['url']
        content = self.content_based.recommend(doc_index, top_k)
        collaborative = self.collaborative.recommend_similar_items(query_url, top_k)
        hybrid = self.hybrid.recommend(doc_index, top_k)

        rows = []
        for rank in range(top_k):
            rows.append({
                'rank': rank + 1,
                'content_based': content[rank]['url'].split('/')[-1] if rank < len(content) else '-',
                'content_score': round(content[rank]['similarity_score'], 4) if rank < len(content) else None,
                'collaborative': collaborative[rank]['url'].split('/')[-1] if rank < len(collaborative) else '-',
                'collab_score': round(collaborative[rank]['similarity_score'], 4) if rank < len(collaborative) else None,
                'hybrid': hybrid[rank]['url'].split('/')[-1] if rank < len(hybrid) else '-',
                'hybrid_score': round(hybrid[rank]['similarity_score'], 4) if rank < len(hybrid) else None,
            })
        return pd.DataFrame(rows)
    
    def get_statistics(self) -> Dict:
        """Return descriptive statistics about the fitted recommenders."""
        document_urls = [doc['url'] for doc in self.documents]
        stats = {
            'approach': self.approach,
            'documents': len(self.documents),
            'content_features': self.content_based.tfidf_matrix.shape[1]
            if self.content_based.tfidf_matrix is not None else 0,
            'collaborative_available': self.collaborative.is_fitted()
        }
        stats.update(self.collaborative.get_coverage(document_urls))
        return stats
    
    def visualize_recommendations(self, recommendations: List[Dict]) -> go.Figure:
        """Visualize recommendations based on approach."""
        if not recommendations:
            return go.Figure(layout=go.Layout(title='No recommendations to display'))

        if self.approach == 'hybrid' and 'content_contribution' in recommendations[0]:
            return self.hybrid.visualize_hybrid_recommendations(recommendations)

        labels = [
            (rec.get('url') or rec.get('item_id', ''))
            for rec in recommendations
        ]
        labels = [label.split('/')[-1][:40] for label in labels]
        scores = [rec.get('similarity_score', rec.get('predicted_rating', 0))
                  for rec in recommendations]

        fig = px.bar(
            x=labels,
            y=scores,
            title=f'Top-{len(recommendations)} Recommendations ({self.approach})',
            labels={'x': 'Recommended item', 'y': 'Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
