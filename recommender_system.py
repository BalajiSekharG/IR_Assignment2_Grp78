import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.vectorizer = None
        self.documents = []
        self.document_urls = []
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
        
        if processed_docs is None:
            from text_preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()
            processed_docs = [preprocessor.preprocess_pipeline(doc['content']) for doc in documents]
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
    
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
        
        # Get indices of top-k similar documents (excluding the query itself)
        top_indices = sim_scores.argsort()[::-1][1:top_k + 1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'url': self.document_urls[idx],
                'title': self.documents[idx].get('title', ''),
                'similarity_score': sim_scores[idx],
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
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarity with all documents
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get indices of top-k similar documents
        top_indices = sim_scores.argsort()[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'url': self.document_urls[idx],
                'title': self.documents[idx].get('title', ''),
                'similarity_score': sim_scores[idx],
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
        
    def fit(self, ratings_data: List[Dict]):
        """
        Fit collaborative filtering model.
        
        Args:
            ratings_data: List of dictionaries with 'user_id', 'item_id', 'rating'
        """
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
        
        # Compute user-user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
    
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
                    'predicted_rating': predicted_rating
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
                    'predicted_rating': predicted_rating
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
    
    def recommend(self, doc_index: int, top_k: int = 5, 
                  content_weight: float = 0.7, collab_weight: float = 0.3) -> List[Dict]:
        """
        Recommend using hybrid approach.
        
        Args:
            doc_index: Index of query document
            top_k: Number of recommendations
            content_weight: Weight for content-based scores
            collab_weight: Weight for collaborative scores
        
        Returns:
            List of hybrid recommendations
        """
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(doc_index, top_k * 2)
        
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add content-based scores
        for rec in content_recs:
            url = rec['url']
            combined_scores[url] = {
                'url': url,
                'title': rec['title'],
                'content_score': rec['similarity_score'],
                'collab_score': 0,
                'combined_score': rec['similarity_score'] * content_weight
            }
        
        # Normalize and combine scores
        max_content = max([r['content_score'] for r in combined_scores.values()]) if combined_scores else 1
        
        for url, rec_data in combined_scores.items():
            rec_data['content_score'] = rec_data['content_score'] / max_content
            rec_data['combined_score'] = rec_data['content_score'] * content_weight
        
        # Sort by combined score
        sorted_recs = sorted(combined_scores.values(), key=lambda x: x['combined_score'], reverse=True)
        
        # Format output
        recommendations = []
        for rec in sorted_recs[:top_k]:
            # Find original document
            doc = next((d for d in self.documents if d['url'] == rec['url']), None)
            if doc:
                recommendations.append({
                    'url': rec['url'],
                    'title': rec['title'],
                    'similarity_score': rec['combined_score'],
                    'content_contribution': rec['content_score'] * content_weight,
                    'collab_contribution': rec['collab_score'] * collab_weight,
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
        
    def fit(self, documents: List[Dict], processed_docs: List[str] = None,
            ratings_data: List[Dict] = None, approach: str = 'content'):
        """
        Fit the recommender system.
        
        Args:
            documents: List of document dictionaries
            processed_docs: Preprocessed documents
            ratings_data: Ratings data for collaborative filtering
            approach: 'content', 'collaborative', or 'hybrid'
        """
        self.approach = approach
        
        if approach == 'content':
            self.content_based.fit(documents, processed_docs)
        elif approach == 'collaborative':
            if ratings_data:
                self.collaborative.fit(ratings_data)
        elif approach == 'hybrid':
            self.hybrid.fit(documents, processed_docs, ratings_data)
    
    def recommend(self, doc_index: int = None, query: str = None, 
                  user_id: str = None, top_k: int = 5) -> List[Dict]:
        """
        Generate recommendations.
        
        Args:
            doc_index: Index of document for content-based
            query: Text query for content-based
            user_id: User ID for collaborative filtering
            top_k: Number of recommendations
        
        Returns:
            List of recommendations
        """
        if self.approach == 'content':
            if query:
                return self.content_based.recommend_by_query(query, top_k)
            elif doc_index is not None:
                return self.content_based.recommend(doc_index, top_k)
        elif self.approach == 'collaborative':
            if user_id:
                return self.collaborative.recommend_user_based(user_id, top_k)
        elif self.approach == 'hybrid':
            if doc_index is not None:
                return self.hybrid.recommend(doc_index, top_k)
        
        return []
    
    def visualize_recommendations(self, recommendations: List[Dict]) -> go.Figure:
        """Visualize recommendations based on approach."""
        if self.approach == 'content':
            return self.content_based.visualize_recommendations(recommendations)
        elif self.approach == 'hybrid':
            return self.hybrid.visualize_hybrid_recommendations(recommendations)
        else:
            # Simple bar chart for collaborative
            items = [rec['item_id'] for rec in recommendations]
            ratings = [rec['predicted_rating'] for rec in recommendations]
            
            fig = px.bar(
                x=list(items),
                y=list(ratings),
                title='Collaborative Filtering Recommendations',
                labels={'x': 'Item ID', 'y': 'Predicted Rating'}
            )
            return fig
