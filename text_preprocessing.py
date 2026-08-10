import nltk
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def safe_df_bounds(n_docs: int) -> Tuple[int, float]:
    """Return (min_df, max_df) that are safe for the size of the collection.

    Two problems are avoided here:
    1. A hard-coded ``min_df=2``/``max_df=0.8`` pair raises
       ``ValueError: max_df corresponds to < documents than min_df`` on tiny
       collections, which is exactly what happens after a shallow crawl.
    2. More subtly, ``min_df=2`` deletes every term that occurs in only one
       document. Those terms are the *most* discriminative ones for retrieval,
       so queries for them silently return zero results. Rare terms are
       therefore kept for any collection small enough for it to matter.
    """
    if n_docs < 50:
        return 1, 1.0
    return 2, 0.95


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
    
    def stem(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    
    def preprocess_pipeline(self, text: str, remove_stops: bool = True,
                           lemmatize_tokens: bool = True, stem_tokens: bool = False) -> str:
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        if lemmatize_tokens:
            tokens = self.lemmatize(tokens)
        if stem_tokens:
            tokens = self.stem(tokens)
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
        self.options = {'remove_stops': True, 'lemmatize_tokens': True, 'stem_tokens': False}
        
    def load_documents(self, documents: List[Dict], remove_stops: bool = True,
                      lemmatize_tokens: bool = True, stem_tokens: bool = False):
        """Load documents and preprocess them with the requested options."""
        self.documents = documents
        self.options = {
            'remove_stops': remove_stops,
            'lemmatize_tokens': lemmatize_tokens,
            'stem_tokens': stem_tokens
        }
        self.processed_docs = []
        for doc in documents:
            processed = self.preprocessor.preprocess_pipeline(
                doc['content'],
                remove_stops=remove_stops,
                lemmatize_tokens=lemmatize_tokens,
                stem_tokens=stem_tokens
            )
            self.processed_docs.append(processed)
        # Feature matrices computed from older documents are now stale.
        self.tfidf_matrix = None
        self.count_matrix = None
        self.document_profiles = []
    
    def extract_tfidf_features(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        """Extract TF-IDF features."""
        min_df, max_df = safe_df_bounds(len(self.processed_docs))
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_docs)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def extract_count_features(self, max_features: int = 5000):
        """Extract count features for LDA."""
        min_df, max_df = safe_df_bounds(len(self.processed_docs))
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df
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
            'doc_index': doc_index,
            'url': doc['url'],
            'word_count': word_count,
            'unique_words': unique_words,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0,
            'top_keywords': keywords
        }
        
        # Replace an existing profile instead of appending duplicates.
        self.document_profiles = [p for p in self.document_profiles
                                   if p['doc_index'] != doc_index]
        self.document_profiles.append(profile)
        return profile
    
    def profile_all_documents(self, top_n: int = 5) -> pd.DataFrame:
        """Profile every document and return a comparable table."""
        rows = []
        for i in range(len(self.documents)):
            profile = self.profile_document(i)
            rows.append({
                'doc_index': i,
                'url': profile['url'],
                'word_count': profile['word_count'],
                'unique_words': profile['unique_words'],
                'char_count': profile['char_count'],
                'avg_word_length': round(profile['avg_word_length'], 2),
                'lexical_diversity': round(profile['lexical_diversity'],
                                             3),
                'top_keywords': ', '.join(term for term, _ in profile['top_keywords'][:top_n])
            })
        return pd.DataFrame(rows)
    
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
        
        # K-means cannot ask for more clusters than there are documents.
        n_clusters = max(2, min(n_clusters, self.tfidf_matrix.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.tfidf_matrix)
        
        # Calculate cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Get top terms for each cluster
        cluster_terms = {}
        for i in range(n_clusters):
            top_indices = cluster_centers[i].argsort()[-10:][::-1]
            top_terms = [self.vocabulary[idx] for idx in top_indices
                        if idx < len(self.vocabulary)]
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
    
    def apply_strategy(self, strategy: str) -> List[str]:
        """Return the corpus processed with one named preprocessing strategy."""
        processed = []
        for doc in self.documents:
            if strategy == 'raw':
                text = doc['content']
            elif strategy == 'basic':
                text = self.preprocessor.clean_text(doc['content'])
            elif strategy == 'tokenize':
                text = ' '.join(self.preprocessor.tokenize(doc['content']))
            elif strategy == 'stopwords':
                tokens = self.preprocessor.tokenize(
                    self.preprocessor.clean_text(doc['content']))
                tokens = self.preprocessor.remove_stopwords(tokens)
                text = ' '.join(tokens)
            elif strategy == 'stemming':
                text = self.preprocessor.preprocess_pipeline(
                    doc['content'], remove_stops=True,
                    lemmatize_tokens=False, stem_tokens=True)
            elif strategy == 'lemmatization':
                text = self.preprocessor.preprocess_pipeline(
                    doc['content'], remove_stops=True,
                    lemmatize_tokens=True, stem_tokens=False)
            elif strategy == 'full':
                text = self.preprocessor.preprocess_pipeline(
                    doc['content'], remove_stops=True,
                    lemmatize_tokens=True, stem_tokens=True)
            else:
                text = doc['content']
            processed.append(text)
        return processed
    
    def compare_preprocessing_strategies(self, strategies: List[str] = None) -> pd.DataFrame:
        """Compare preprocessing strategies on size and compression.

        Returns a table so the effect of each stage on vocabulary size (and
        therefore on index size) can be reported directly in the front end.
        """
        if strategies is None:
            strategies = ['raw', 'basic', 'stopwords', 'lemmatization', 'stemming', 'full']
        
        rows = []
        baseline_vocab = None
        
        for strategy in strategies:
            processed = self.apply_strategy(strategy)
            total_words = sum(len(text.split()) for text in processed)
            vocabulary = set(' '.join(processed).split())
            avg_length = total_words / len(processed) if processed else 0
            
            if baseline_vocab is None:
                baseline_vocab = vocabulary
            
            rows.append({
                'strategy': strategy,
                'total_words': total_words,
                'vocabulary_size': len(vocabulary),
                'avg_doc_length': round(avg_length, 2),
                'vocab_reduction_pct': round((1 - len(vocabulary) / len(baseline_vocab)) * 100, 2) if baseline_vocab else 0
            })
        
        return pd.DataFrame(rows)
    
    def compare_feature_extraction(self, labels: List[str] = None) -> pd.DataFrame:
        """Compare feature extraction configurations.

        When labels are supplied each representation is also scored with
        a cross-validated classifier, which turns the comparison into a
        quantitative rather than a descriptive one.
        """
        min_df, max_df = safe_df_bounds(len(self.processed_docs))
        configurations = {
            'count_unigram': CountVectorizer(min_df=min_df, max_df=max_df),
            'tfidf_unigram': TfidfVectorizer(min_df=min_df, max_df=max_df),
            'tfidf_bigram': TfidfVectorizer(min_df=min_df, max_df=max_df,
                                           ngram_range=(1, 2)),
            'tfidf_sublinear': TfidfVectorizer(min_df=min_df, max_df=max_df,
                                              sublinear_tf=True),
        }
        
        rows = []
        for name, vectorizer in configurations.items():
            matrix = vectorizer.fit_transform(self.processed_docs)
            density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[1] else 0
            row = {
                'representation': name,
                'n_features': matrix.shape[1],
                'non_zero_entries': int(matrix.nnz),
                'matrix_density': round(density, 4)
            }
            
            if labels is not None and len(set(labels)) > 1:
                folds = min(3, min(Counter(labels).values()))
                if folds >= 2:
                    scores = cross_val_score(
                        LogisticRegression(max_iter=1000),
                        matrix, labels, cv=folds, scoring='accuracy')
                    row['cv_accuracy'] = round(float(np.mean(scores)), 3)
                else:
                    row['cv_accuracy'] = float('nan')
            else:
                row['cv_accuracy'] = float('nan')
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def classify_documents(self, labels: List[str], model_name: str = 'logistic_regression',
                          test_size: float = 0.3) -> Dict:
        """Train and evaluate a supervised document classifier.

        Args:
            labels: One category label per document.
            model_name: 'logistic_regression', 'naive_bayes' or 'linear_svm'.
            test_size: Fraction of documents held out for testing.

        Returns:
            Dictionary with fitted model, held-out metrics and confusion
            matrix.
        """
        if len(labels) != len(self.processed_docs):
            raise ValueError(
                f"{len(labels)} labels supplied for {len(self.processed_docs)} documents")
        
        if self.tfidf_matrix is None:
            self.extract_tfidf_features()
        
        features = self.tfidf_matrix
        label_counts = Counter(labels)
        stratify = labels if min(label_counts.values()) >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=stratify)
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'linear_svm': LinearSVC()
        }
        model = models.get(model_name, models['logistic_regression'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted', zero_division=0)
        
        class_names = sorted(set(labels))
        matrix = confusion_matrix(y_test, predictions, labels=class_names)
        
        folds = min(3, min(label_counts.values()))
        cv_accuracy = float('nan')
        if folds >= 2:
            cv_accuracy = float(np.mean(cross_val_score(
                model, features, labels, cv=folds, scoring='accuracy')))
        
        return {
            'model_name': model_name,
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_accuracy': cv_accuracy,
            'class_names': class_names,
            'confusion_matrix': matrix,
            'report': classification_report(y_test, predictions, zero_division=0),
            'y_test': list(y_test),
            'predictions': list(predictions)
        }
    
    def compare_classifiers(self, labels: List[str]) -> pd.DataFrame:
        """Compare several classifiers on the same feature representation."""
        rows = []
        for model_name in ['logistic_regression', 'naive_bayes', 'linear_svm']:
            try:
                result = self.classify_documents(labels, model_name=model_name)
            except Exception as error:  # keep the UI responsive on bad input
                rows.append({'model': model_name, 'error': str(error)})
                continue
            
            rows.append({
                'model': model_name,
                'accuracy': round(result['accuracy'], 3),
                'precision': round(result['precision'], 3),
                'recall': round(result['recall'], 3),
                'f1_score': round(result['f1_score'], 3),
                'cv_accuracy': round(result['cv_accuracy'], 3)
            })
        return pd.DataFrame(rows)
    
    def visualize_keywords(self, doc_index: int, top_n: int = 10) -> go.Figure:
        """Visualize the top TF-IDF keywords of one document."""
        keywords = self.extract_keywords(doc_index, top_n=top_n)
        keywords = [(term, score) for term, score in keywords if score > 0]
        if not keywords:
            return go.Figure()
        
        terms, scores = zip(*keywords[::-1])
        fig = px.bar(
            x=list(scores),
            y=list(terms),
            orientation='h',
            title=f'Top {len(terms)} TF-IDF Keywords (document {doc_index})',
            labels={'x': 'TF-IDF weight', 'y': 'Term'}
        )
        return fig
    
    def visualize_strategy_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Visualize vocabulary size across preprocessing strategies."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df['strategy'],
            y=comparison_df['vocabulary_size'],
            name='Vocabulary size'
        ))
        fig.add_trace(go.Scatter(
            x=comparison_df['strategy'],
            y=comparison_df['avg_doc_length'],
            name='Avg document length',
            mode='lines+markers',
            yaxis='y2'
        ))
        fig.update_layout(
            title='Effect of Preprocessing Strategy on the Feature Space',
            yaxis=dict(title='Vocabulary size'),
            yaxis2=dict(title='Avg document length', overlaying='y', side='right'),
            xaxis_tickangle=-45
        )
        return fig
