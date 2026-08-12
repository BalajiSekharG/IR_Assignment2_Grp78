import nltk
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    def __init__(self):
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        # Try newer punkt_tab first, then fall back to punkt
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Download wordnet for lemmatization
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return nltk.word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str], custom_stopwords: List[str] = None) -> List[str]:
        """Remove stopwords from tokens."""
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_pipeline(self, text: str, remove_stops: bool = True, 
                           lemmatize_tokens: bool = True) -> str:
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        if lemmatize_tokens:
            tokens = self.lemmatize(tokens)
        return ' '.join(tokens)


class TextMiningFramework:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.documents = []
        self.processed_docs = []
        self.tfidf_matrix = None
        self.count_matrix = None
        self.vectorizer = None
        self.count_vectorizer = None
        self.vocabulary = None
        self.document_profiles = []
        
    def load_documents(self, documents: List[Dict]):
        """Load documents for processing."""
        self.documents = documents
        self.processed_docs = []
        for doc in documents:
            processed = self.preprocessor.preprocess_pipeline(doc['content'])
            self.processed_docs.append(processed)
    
    def extract_tfidf_features(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        """Extract TF-IDF features."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_docs)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def extract_count_features(self, max_features: int = 5000):
        """Extract count features for LDA."""
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8
        )
        self.count_matrix = self.count_vectorizer.fit_transform(self.processed_docs)
        return self.count_matrix
    
    def extract_keywords(self, doc_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords for a document using TF-IDF."""
        if self.tfidf_matrix is None:
            self.extract_tfidf_features()
        
        doc_vector = self.tfidf_matrix[doc_index].toarray()[0]
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        
        keywords = [(self.vocabulary[i], doc_vector[i]) for i in top_indices]
        return keywords
    
    def profile_document(self, doc_index: int) -> Dict:
        """Create a comprehensive profile for a document."""
        doc = self.documents[doc_index]
        processed = self.processed_docs[doc_index]
        
        # Basic statistics
        word_count = len(processed.split())
        unique_words = len(set(processed.split()))
        
        # Keywords
        keywords = self.extract_keywords(doc_index, top_n=10)
        
        # Character statistics
        char_count = len(doc['content'])
        avg_word_length = sum(len(word) for word in processed.split()) / word_count if word_count > 0 else 0
        
        profile = {
            'url': doc['url'],
            'word_count': word_count,
            'unique_words': unique_words,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0,
            'top_keywords': keywords
        }
        
        self.document_profiles.append(profile)
        return profile
    
    def perform_topic_modeling(self, n_topics: int = 5, n_words: int = 10) -> List[List[str]]:
        """Perform LDA topic modeling."""
        if self.count_matrix is None:
            self.extract_count_features()
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(self.count_matrix)
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        
        return topics
    
    def cluster_documents(self, n_clusters: int = 3) -> Dict:
        """Cluster documents using K-means."""
        if self.tfidf_matrix is None:
            self.extract_tfidf_features()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.tfidf_matrix)
        
        # Calculate cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Get top terms for each cluster
        cluster_terms = {}
        for i in range(n_clusters):
            top_indices = cluster_centers[i].argsort()[-10:][::-1]
            top_terms = [self.vocabulary[idx] for idx in top_indices]
            cluster_terms[i] = top_terms
        
        return {
            'clusters': clusters,
            'cluster_terms': cluster_terms,
            'cluster_centers': cluster_centers
        }
    
    def get_corpus_statistics(self) -> Dict:
        """Get comprehensive corpus statistics."""
        total_docs = len(self.documents)
        total_words = sum(len(doc.split()) for doc in self.processed_docs)
        avg_doc_length = total_words / total_docs if total_docs > 0 else 0
        
        # Vocabulary size
        all_words = ' '.join(self.processed_docs).split()
        vocab_size = len(set(all_words))
        
        # Document length distribution
        doc_lengths = [len(doc.split()) for doc in self.processed_docs]
        
        return {
            'total_documents': total_docs,
            'total_words': total_words,
            'vocabulary_size': vocab_size,
            'average_document_length': avg_doc_length,
            'min_document_length': min(doc_lengths) if doc_lengths else 0,
            'max_document_length': max(doc_lengths) if doc_lengths else 0,
            'document_length_distribution': doc_lengths
        }
    
    def get_most_frequent_words(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Get most frequent words in corpus."""
        all_words = ' '.join(self.processed_docs).split()
        word_freq = Counter(all_words)
        return word_freq.most_common(top_n)
    
    def visualize_document_lengths(self) -> go.Figure:
        """Create visualization for document length distribution."""
        stats = self.get_corpus_statistics()
        
        fig = px.histogram(
            x=stats['document_length_distribution'],
            nbins=30,
            title='Document Length Distribution',
            labels={'x': 'Document Length (words)', 'y': 'Frequency'}
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def visualize_vocabulary_distribution(self) -> go.Figure:
        """Create visualization for top vocabulary words."""
        freq_words = self.get_most_frequent_words(20)
        words, counts = zip(*freq_words)
        
        fig = px.bar(
            x=list(words),
            y=list(counts),
            title='Top 20 Most Frequent Words',
            labels={'x': 'Words', 'y': 'Frequency'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def visualize_topic_distribution(self, topics: List[List[str]]) -> go.Figure:
        """Create visualization for topic modeling results."""
        topic_names = [f'Topic {i+1}' for i in range(len(topics))]
        
        fig = go.Figure()
        for i, topic in enumerate(topics):
            words = topic[:5]
            weights = list(range(len(words), 0, -1))
            fig.add_trace(go.Bar(
                name=topic_names[i],
                x=words,
                y=weights,
                orientation='v'
            ))
        
        fig.update_layout(
            title='Topic Modeling Results (Top 5 Words per Topic)',
            barmode='group',
            xaxis_tickangle=-45
        )
        return fig
    
    def visualize_clusters(self, cluster_result: Dict) -> go.Figure:
        """Create visualization for document clusters."""
        if self.tfidf_matrix is None:
            self.extract_tfidf_features()
        
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        doc_vectors_2d = pca.fit_transform(self.tfidf_matrix.toarray())
        
        clusters = cluster_result['clusters']
        
        fig = px.scatter(
            x=doc_vectors_2d[:, 0],
            y=doc_vectors_2d[:, 1],
            color=clusters,
            title='Document Clusters (PCA Visualization)',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'}
        )
        return fig
    
    def compare_preprocessing_strategies(self, strategies: List[str]) -> Dict:
        """Compare different preprocessing strategies."""
        results = {}
        
        for strategy in strategies:
            processed = []
            for doc in self.documents:
                if strategy == 'basic':
                    text = self.preprocessor.clean_text(doc['content'])
                elif strategy == 'tokenize':
                    text = ' '.join(self.preprocessor.tokenize(doc['content']))
                elif strategy == 'stopwords':
                    tokens = self.preprocessor.tokenize(doc['content'])
                    tokens = self.preprocessor.remove_stopwords(tokens)
                    text = ' '.join(tokens)
                elif strategy == 'full':
                    text = self.preprocessor.preprocess_pipeline(doc['content'])
                else:
                    text = doc['content']
                processed.append(text)
            
            # Calculate statistics
            total_words = sum(len(doc.split()) for doc in processed)
            unique_words = len(set(' '.join(processed).split()))
            avg_length = total_words / len(processed) if processed else 0
            
            results[strategy] = {
                'total_words': total_words,
                'vocabulary_size': unique_words,
                'avg_document_length': avg_length
            }
        
        return results
