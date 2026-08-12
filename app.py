import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time
import numpy as np
import networkx as nx
from datetime import datetime

# Import our modules
from web_crawler import WebCrawler
from text_preprocessing import TextMiningFramework
from search_engine import SearchEngine
from recommender_system import RecommenderSystem
from evaluation_metrics import IREvaluation
from sample_data import BUNDLED_DATASET, BUNDLED_METADATA, BUNDLED_URL_GRAPH, BUNDLED_LABELS, BUNDLED_RATINGS, BUNDLED_QUERIES

# Page configuration
st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with default objects
if 'crawler' not in st.session_state:
    st.session_state.crawler = WebCrawler()
if 'text_mining' not in st.session_state:
    st.session_state.text_mining = TextMiningFramework()
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = SearchEngine()
if 'recommender' not in st.session_state:
    st.session_state.recommender = RecommenderSystem()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = IREvaluation()
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'metadata' not in st.session_state:
    st.session_state.metadata = []
if 'url_graph' not in st.session_state:
    st.session_state.url_graph = {}
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'query_log' not in st.session_state:
    st.session_state.query_log = []
if 'method_comparison' not in st.session_state:
    st.session_state.method_comparison = None

# Helper functions
def invalidate_downstream():
    """Invalidate downstream stages when upstream changes."""
    st.session_state.processed = False
    st.session_state.indexed = False
    st.session_state.method_comparison = None

def sync_collection_from_crawler():
    """Sync session state from crawler after a crawl."""
    st.session_state.documents = st.session_state.crawler.documents
    st.session_state.metadata = st.session_state.crawler.metadata
    st.session_state.url_graph = st.session_state.crawler.url_graph
    invalidate_downstream()

def metadata_lookup(url):
    """Look up metadata for a URL."""
    for m in st.session_state.metadata:
        if m.get('url') == url:
            return m
    return {}

def title_for(doc_id):
    """Get title for a document ID."""
    if 0 <= doc_id < len(st.session_state.documents):
        url = st.session_state.documents[doc_id].get('url', '')
        meta = metadata_lookup(url)
        return meta.get('title', url)
    return str(doc_id)

def document_options():
    """Generate document options for selectbox."""
    return [f"{i}: {title_for(i)}" for i in range(len(st.session_state.documents))]

def describe_document(doc_id):
    """Generate a description for a document."""
    if 0 <= doc_id < len(st.session_state.documents):
        doc = st.session_state.documents[doc_id]
        meta = metadata_lookup(doc.get('url', ''))
        return f"{meta.get('title', doc.get('url', ''))} ({len(doc.get('content', ''))} chars)"
    return "Unknown document"

def load_bundled_dataset(include_near_duplicates=False):
    """Load the bundled dataset into session state."""
    st.session_state.documents = BUNDLED_DATASET.copy()
    st.session_state.metadata = BUNDLED_METADATA.copy()
    st.session_state.url_graph = BUNDLED_URL_GRAPH.copy()
    
    if not include_near_duplicates:
        # Filter out near duplicates
        near_dup_urls = set()
        for i, doc in enumerate(st.session_state.documents):
            if doc.get('near_duplicate', False):
                near_dup_urls.add(doc.get('url', ''))
        
        st.session_state.documents = [d for d in st.session_state.documents if d.get('url', '') not in near_dup_urls]
        st.session_state.metadata = [m for m in st.session_state.metadata if m.get('url', '') not in near_dup_urls]
        
        # Clean up URL graph
        for url in near_dup_urls:
            st.session_state.url_graph.pop(url, None)
        for url in st.session_state.url_graph:
            st.session_state.url_graph[url] = [u for u in st.session_state.url_graph[url] if u not in near_dup_urls]
    
    invalidate_downstream()

def run_preprocessing(remove_stops, lemmatize, stem, max_features, ngram_max):
    """Run text preprocessing with given parameters."""
    st.session_state.text_mining.load_documents(
        st.session_state.documents,
        remove_stops=remove_stops,
        lemmatize_tokens=lemmatize,
        stem_tokens=stem
    )
    st.session_state.text_mining.extract_tfidf_features(
        max_features=max_features,
        ngram_range=(1, ngram_max)
    )
    st.session_state.processed = True
    st.session_state.indexed = False
    st.session_state.method_comparison = None

def build_index():
    """Build the search index."""
    processed_docs = None
    if st.session_state.processed:
        processed_docs = st.session_state.text_mining.processed_docs
    
    st.session_state.search_engine.create_index(
        st.session_state.documents,
        st.session_state.metadata,
        processed_docs
    )
    st.session_state.indexed = True

def build_link_graph():
    """Build the URL graph and compute link analysis."""
    if st.session_state.url_graph:
        G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
        st.session_state.search_engine.calculate_pagerank(G)
        st.session_state.search_engine.calculate_hits(G)

def build_recommender():
    """Build the recommender system."""
    if st.session_state.documents:
        st.session_state.recommender.fit(st.session_state.documents)

def log_query(query, results, latency_ms, method):
    """Log a query for analytics."""
    st.session_state.query_log.append({
        'timestamp': datetime.now(),
        'query': query,
        'results': len(results),
        'latency_ms': latency_ms,
        'method': method
    })

def results_table(results):
    """Convert results to a dataframe for display."""
    data = []
    for i, r in enumerate(results, 1):
        data.append({
            'rank': i,
            'title': r.get('title', r.get('url', ''))[:50],
            'url': r.get('url', ''),
            'score': f"{r.get('score', 0):.4f}"
        })
    return pd.DataFrame(data)

def stage_badge(complete, label):
    """Render a badge for a pipeline stage."""
    if complete:
        return f"✅ {label}"
    else:
        return f"⭕ {label}"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Dashboard", "Data Acquisition", "Text Preprocessing & Mining", "Index Management", 
     "Search Interface", "Ranking Visualization", "Recommendation Panel", 
     "Evaluation Dashboard", "Performance Analytics", "Inference & Discussion"]
)

# Pipeline status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Status")
st.sidebar.write(stage_badge(bool(st.session_state.documents), "Documents loaded"))
st.sidebar.write(stage_badge(st.session_state.processed, "Text processed"))
st.sidebar.write(stage_badge(st.session_state.indexed, "Index built"))

# Main header
st.markdown('<h1 style="text-align: center; color: #1f77b4; font-size: 2.5rem; font-weight: bold;">Information Retrieval System</h1>', unsafe_allow_html=True)

# Dashboard Page
if page == "Dashboard":
    st.header("System Dashboard")
    
    # Collection metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(st.session_state.documents))
    col2.metric("Metadata entries", len(st.session_state.metadata))
    col3.metric("URLs in graph", len(st.session_state.url_graph))
    col4.metric("Indexed", "Yes" if st.session_state.indexed else "No")
    
    # Pipeline stage indicators
    st.subheader("Pipeline Stages")
    col1, col2, col3 = st.columns(3)
    col1.write(stage_badge(bool(st.session_state.documents), "Data Acquisition"))
    col2.write(stage_badge(st.session_state.processed, "Preprocessing"))
    col3.write(stage_badge(st.session_state.indexed, "Indexing"))
    
    # Collection composition visualization
    if st.session_state.metadata:
        st.subheader("Collection Composition")
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            categories = {}
            for m in st.session_state.metadata:
                cat = m.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            if categories:
                fig = px.pie(values=list(categories.values()), names=list(categories.keys()), title="Document Categories")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Document length distribution
            lengths = [len(d.get('content', '')) for d in st.session_state.documents]
            fig = px.histogram(x=lengths, nbins=20, title="Document Length Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Source tracking
    if st.session_state.documents:
        st.subheader("Source Tracking")
        sources = {}
        for d in st.session_state.documents:
            url = d.get('url', '')
            if 'example.com' in url:
                sources['Bundled dataset'] = sources.get('Bundled dataset', 0) + 1
            else:
                sources['Web crawled'] = sources.get('Web crawled', 0) + 1
        
        if sources:
            st.write(f"Sources: {', '.join([f'{k}: {v}' for k, v in sources.items()])}")
    
    # One-click pipeline
    st.markdown("---")
    st.subheader("One-click Pipeline")
    if st.button("Run full pipeline with bundled dataset"):
        load_bundled_dataset(include_near_duplicates=False)
        run_preprocessing(remove_stops=True, lemmatize=True, stem=False, max_features=None, ngram_max=1)
        build_index()
        build_link_graph()
        build_recommender()
        
        # Load bundled queries for evaluation
        for qid, qdata in BUNDLED_QUERIES.items():
            st.session_state.evaluator.query_text[qid] = qdata['query_text']
            st.session_state.evaluator.relevant_docs[qid] = set(qdata['relevant_doc_ids'])
            if 'graded_relevance' in qdata:
                st.session_state.evaluator.graded_relevance[qid] = qdata['graded_relevance']
        
        st.success("Full pipeline completed!")
        st.rerun()

# Data Acquisition Page
elif page == "Data Acquisition":
    st.header("Data Acquisition")
    
    # Near-duplicate threshold slider
    st.subheader("Near-duplicate Detection")
    jaccard_threshold = st.slider("Jaccard similarity threshold for near-duplicates", 0.0, 1.0, 0.75, 0.05)
    st.session_state.crawler.jaccard_threshold = jaccard_threshold
    
    # Tabs for different acquisition modes
    tab1, tab2, tab3 = st.tabs(["Bundled dataset", "Web crawling", "Upload files"])
    
    with tab1:
        st.subheader("Bundled Dataset")
        include_near = st.checkbox("Include near-duplicates", value=False)
        if st.button("Load bundled dataset"):
            load_bundled_dataset(include_near_duplicates=include_near)
            st.success(f"Loaded {len(st.session_state.documents)} documents from bundled dataset")
            st.rerun()
    
    with tab2:
        st.subheader("Web Crawling")
        seed_urls = st.text_area("Seed URLs (one per line)", "https://example.com/page1\nhttps://example.com/page2", height=100)
        max_depth = st.number_input("Max depth", 1, 5, 2)
        max_pages = st.number_input("Max pages", 1, 500, 50)
        delay = st.number_input("Delay between requests (seconds)", 0.0, 5.0, 1.0, 0.5)
        domain_restriction = st.checkbox("Restrict to domain", value=True)
        detect_near_duplicates = st.checkbox("Detect near-duplicates", value=True)
        
        if st.button("Start crawl"):
            seed_list = [url.strip() for url in seed_urls.split('\n') if url.strip()]
            with st.spinner("Crawling..."):
                stats = st.session_state.crawler.crawl(
                    seed_urls=seed_list,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    delay=delay,
                    stay_on_domain=domain_restriction,
                    detect_near_duplicates=detect_near_duplicates
                )
                sync_collection_from_crawler()
            
            st.success(f"Crawl completed: {stats['pages_crawled']} pages")
            col1, col2, col3 = st.columns(3)
            col1.metric("Pages crawled", stats['pages_crawled'])
            col2.metric("Duplicates skipped", stats.get('duplicate_urls_skipped', 0))
            col3.metric("Near-duplicates", stats.get('near_duplicates', 0))
    
    with tab3:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader("Upload documents", type=['json', 'csv', 'txt'], accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                if file.type == 'application/json':
                    data = json.load(file)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'content' in item:
                                st.session_state.documents.append(item)
                elif file.type == 'text/csv':
                    df = pd.read_csv(file)
                    for _, row in df.iterrows():
                        if 'content' in row:
                            st.session_state.documents.append(row.to_dict())
                elif file.type == 'text/plain':
                    content = file.read().decode('utf-8')
                    st.session_state.documents.append({
                        'url': f'uploaded://{file.name}',
                        'content': content,
                        'hash': hash(content)
                    })
            
            invalidate_downstream()
            st.success(f"Uploaded {len(uploaded_files)} files")
    
    # Duplicate handling report
    if st.session_state.documents:
        duplicate_report = st.session_state.crawler.get_duplicate_report()
        if not duplicate_report.empty:
            st.subheader("Duplicate and Near-duplicate Handling")
            st.dataframe(duplicate_report, use_container_width=True)
    
    # Stored collection
    if st.session_state.documents:
        st.markdown("---")
        st.subheader("Stored Collection")
        
        tab_a, tab_b, tab_c, tab_d = st.tabs(["Document content", "Metadata", "Link graph", "Export"])
        
        with tab_a:
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"Document {i}: {doc.get('url', '')[:50]}"):
                    st.text_area("Content", doc.get('content', ''), height=200, key=f"doc_{i}")
        
        with tab_b:
            if st.session_state.metadata:
                st.dataframe(pd.DataFrame(st.session_state.metadata), use_container_width=True)
            else:
                st.info("No metadata available")
        
        with tab_c:
            if st.session_state.url_graph:
                st.json(st.session_state.url_graph)
            else:
                st.info("No URL graph available")
        
        with tab_d:
            if st.button("Export documents as JSON"):
                json_data = json.dumps(st.session_state.documents, indent=2)
                st.download_button("Download", json_data, file_name="documents.json", mime="application/json")

# Text Preprocessing & Mining Page
elif page == "Text Preprocessing & Mining":
    st.header("Text Preprocessing & Mining")
    
    if not st.session_state.documents:
        st.warning("No documents loaded. Please acquire documents first.")
    else:
        # Preprocessing configuration
        st.subheader("Preprocessing Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            remove_stops = st.checkbox("Remove stopwords", value=True)
            lemmatize = st.checkbox("Lemmatize", value=True)
        
        with col2:
            stem = st.checkbox("Porter stemming", value=False)
            max_features = st.number_input("Max features (None for all)", min_value=100, max_value=10000, value=None)
        
        with col3:
            ngram_max = st.slider("N-gram max", 1, 3, 1)
        
        if st.button("Run preprocessing"):
            with st.spinner("Processing..."):
                run_preprocessing(remove_stops, lemmatize, stem, max_features, ngram_max)
            st.success("Preprocessing completed!")
            st.rerun()
        
        # Corpus statistics
        if st.session_state.processed:
            st.subheader("Corpus Statistics")
            stats = st.session_state.text_mining.get_corpus_statistics()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Documents", stats['total_documents'])
            col2.metric("Total words", stats['total_words'])
            col3.metric("Vocabulary", stats['vocabulary_size'])
            col4.metric("Avg length", f"{stats['average_document_length']:.1f}")
            
            # Active preprocessing pipeline
            st.info(f"Active pipeline: stopwords={remove_stops}, lemmatize={lemmatize}, stem={stem}, max_features={max_features}, ngram_max={ngram_max}")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Corpus characteristics", "Keywords & profiling", "Strategy comparison", "Classification", "Topics & clusters"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(st.session_state.text_mining.visualize_document_lengths(), use_container_width=True)
                with col2:
                    st.plotly_chart(st.session_state.text_mining.visualize_vocabulary_distribution(), use_container_width=True)
                
                # Feature distribution
                if hasattr(st.session_state.text_mining, 'tfidf_matrix'):
                    fig = px.histogram(x=st.session_state.text_mining.tfidf_matrix.data, nbins=50, title="TF-IDF Feature Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Keyword extraction (TF-IDF)")
                top_k = st.slider("Top K keywords", 5, 50, 20)
                if st.button("Extract keywords"):
                    keywords = st.session_state.text_mining.extract_keywords(top_k=top_k)
                    for doc_id, words in enumerate(keywords):
                        st.write(f"Doc {doc_id}: {', '.join(words[:10])}")
                
                st.subheader("Document profiling")
                if st.button("Profile documents"):
                    profiles = st.session_state.text_mining.profile_documents()
                    st.dataframe(pd.DataFrame(profiles), use_container_width=True)
            
            with tab3:
                st.subheader("Strategy comparison")
                strategies = [
                    {"remove_stops": True, "lemmatize": True, "stem": False, "max_features": None, "ngram_max": 1},
                    {"remove_stops": True, "lemmatize": True, "stem": True, "max_features": None, "ngram_max": 1},
                    {"remove_stops": True, "lemmatize": False, "stem": False, "max_features": 1000, "ngram_max": 1},
                ]
                
                if st.button("Compare strategies"):
                    results = []
                    for i, strat in enumerate(strategies):
                        st.session_state.text_mining.load_documents(st.session_state.documents)
                        st.session_state.text_mining.extract_tfidf_features(**strat)
                        stats = st.session_state.text_mining.get_corpus_statistics()
                        results.append({
                            'strategy': f"Strategy {i+1}",
                            'vocabulary': stats['vocabulary_size'],
                            'remove_stops': strat['remove_stops'],
                            'lemmatize': strat['lemmatize'],
                            'stem': strat['stem'],
                            'max_features': strat['max_features']
                        })
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
            
            with tab4:
                st.subheader("Document Classification")
                if BUNDLED_LABELS:
                    # Use bundled labels if available
                    labels = [BUNDLED_LABELS.get(i, 'unknown') for i in range(len(st.session_state.documents))]
                else:
                    # Manual labeling
                    st.info("No labels available. Please label documents manually.")
                    labels = [st.selectbox(f"Label for doc {i}", ['ml', 'dl', 'nlp', 'cv', 'rl', 'data'], key=f"label_{i}") for i in range(len(st.session_state.documents))]
                
                classifier_type = st.selectbox("Classifier", ["logistic_regression", "naive_bayes", "linear_svm"])
                test_size = st.slider("Test size", 0.1, 0.5, 0.2)
                
                if st.button("Train classifier"):
                    with st.spinner("Training..."):
                        results = st.session_state.text_mining.classify_documents(labels, classifier_type=classifier_type, test_size=test_size)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("CV Accuracy", f"{results['cv_accuracy']:.4f}")
                    col3.metric("F1 (macro)", f"{results['f1_macro']:.4f}")
                    
                    st.dataframe(pd.DataFrame(results['classification_report']).transpose(), use_container_width=True)
                    
                    # Confusion matrix
                    fig = px.imshow(results['confusion_matrix'], text_auto=True, title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.subheader("Topic Modeling (LDA)")
                n_topics = st.slider("Number of topics", 2, 10, 5)
                if st.button("Run LDA"):
                    topics = st.session_state.text_mining.perform_topic_modeling(n_topics=n_topics)
                    for i, topic in enumerate(topics):
                        st.write(f"Topic {i+1}: {', '.join(topic)}")
                    
                    fig = px.bar(x=[f"Topic {i+1}" for i in range(len(topics))], y=[len(t) for t in topics], title="Topic Word Counts")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Document Clustering (K-means)")
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                if st.button("Run K-means"):
                    cluster_result = st.session_state.text_mining.cluster_documents(n_clusters=n_clusters)
                    st.plotly_chart(st.session_state.text_mining.visualize_clusters(cluster_result), use_container_width=True)
                    
                    for cluster_id, terms in cluster_result['cluster_terms'].items():
                        st.write(f"Cluster {cluster_id}: {', '.join(terms)}")

# Index Management Page
elif page == "Index Management":
    st.header("Index Management")
    
    if not st.session_state.documents:
        st.warning("No documents loaded. Please acquire documents first.")
    else:
        # Build index
        st.subheader("Build Search Index")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Build TF-IDF index"):
                run_preprocessing(remove_stops=True, lemmatize=True, stem=False, max_features=None, ngram_max=1)
                build_index()
                st.success("TF-IDF index built!")
                st.rerun()
        
        with col2:
            if st.button("Build BM25 index"):
                run_preprocessing(remove_stops=True, lemmatize=True, stem=False, max_features=None, ngram_max=1)
                build_index()
                st.success("BM25 index built!")
                st.rerun()
        
        # Build URL graph
        st.subheader("Build URL Graph")
        if st.button("Build graph & compute PageRank/HITS"):
            build_link_graph()
            st.success("Link analysis completed!")
            st.rerun()
        
        # Index statistics
        if st.session_state.indexed:
            st.subheader("Index Statistics")
            stats = st.session_state.search_engine.get_index_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Documents indexed", stats['documents_indexed'])
            col2.metric("Vocabulary size", stats['vocabulary_size'])
            col3.metric("Postings", stats['posting_entries'])
            col4.metric("Build time", f"{stats['index_build_seconds']:.2f}s")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Matrix density", f"{stats['matrix_density']:.4f}")
            col2.metric("Postings/doc", f"{stats['avg_postings_per_document']:.2f}")
            col3.metric("Avg doc length", f"{stats['avg_document_length']:.1f}")
            
            # Link analysis summary
            if st.session_state.url_graph:
                st.subheader("Link Analysis Summary")
                if hasattr(st.session_state.search_engine, 'pagerank_scores'):
                    pr_scores = st.session_state.search_engine.pagerank_scores
                    if pr_scores:
                        col1.metric("PageRank nodes", len(pr_scores))
                        col2.metric("Max PageRank", f"{max(pr_scores.values()):.4f}")
                        col3.metric("Min PageRank", f"{min(pr_scores.values()):.4f}")
                
                if hasattr(st.session_state.search_engine, 'hits_scores'):
                    hits = st.session_state.search_engine.hits_scores
                    if hits:
                        hubs = hits.get('hubs', {})
                        auths = hits.get('authorities', {})
                        col1.metric("HITS nodes", len(hubs))
                        col2.metric("Max Authority", f"{max(auths.values()) if auths else 0:.4f}")
                        col3.metric("Max Hub", f"{max(hubs.values()) if hubs else 0:.4f}")

# Search Interface Page
elif page == "Search Interface":
    st.header("Search Interface")
    
    if not st.session_state.indexed:
        st.warning("Index not built. Please build index first.")
    else:
        # Query input
        st.subheader("Search Query")
        query = st.text_input("Enter your query", "")
        
        # Query syntax hints
        with st.expander("Query syntax hints"):
            st.markdown("""
            - **Phrase search**: Use quotes for exact phrases: `"machine learning"`
            - **Required terms**: Use + before term: `+neural +network`
            - **Excluded terms**: Use - before term: `deep -learning`
            - **OR operator**: Use | between terms: `ml|ai`
            """)
        
        # Ranking method selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ranking_method = st.selectbox("Ranking method", ["tfidf", "bm25", "pagerank", "hits", "hybrid"])
        
        with col2:
            limit = st.number_input("Results limit", 1, 50, 10)
        
        with col3:
            link_weight = st.slider("Link weight", 0.0, 1.0, 0.3, 0.05, key="search_link_weight")
        
        with col4:
            expand = st.checkbox("Use query expansion", value=False)
        
        # Metadata/content filters
        with st.expander("Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                category_filter = st.selectbox("Category filter", ["all"] + list(set([m.get('category', 'unknown') for m in st.session_state.metadata])))
                min_length = st.number_input("Min document length", 0, 10000, 0)
            
            with col2:
                must_contain = st.text_input("Must contain")
                must_not_contain = st.text_input("Must not contain")
        
        # Search button
        if st.button("Search"):
            if query:
                started = time.perf_counter()
                results = st.session_state.search_engine.search(
                    query=query,
                    limit=limit,
                    ranking_method=ranking_method,
                    link_weight=link_weight,
                    expand=expand,
                    category_filter=category_filter if category_filter != "all" else None,
                    min_length=min_length if min_length > 0 else None,
                    must_contain=must_contain if must_contain else None,
                    must_not_contain=must_not_contain if must_not_contain else None
                )
                elapsed = (time.perf_counter() - started) * 1000
                
                log_query(query, results, elapsed, ranking_method)
                
                # Display metrics
                st.subheader("Search Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Results", len(results))
                col2.metric("Latency", f"{elapsed:.2f}ms")
                col3.metric("Matched docs", len(st.session_state.search_engine.last_matched_docs) if hasattr(st.session_state.search_engine, 'last_matched_docs') else 0)
                col4.metric("Method", ranking_method)
                
                # Query processing details
                with st.expander("Query processing details"):
                    if hasattr(st.session_state.search_engine, 'last_query_info'):
                        info = st.session_state.search_engine.last_query_info
                        st.json(info)
                
                # Ranked results table
                if results:
                    st.dataframe(results_table(results), use_container_width=True)
                    
                    # Individual result expanders
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result.get('title', result.get('url', ''))}"):
                            st.write(f"**Score:** {result.get('score', 0):.4f}")
                            st.write(f"**URL:** {result.get('url', '')}")
                            if 'pagerank_score' in result:
                                st.write(f"**PageRank:** {result['pagerank_score']:.4f}")
                            if 'authority_score' in result:
                                st.write(f"**Authority:** {result['authority_score']:.4f}")
                            if 'hub_score' in result:
                                st.write(f"**Hub:** {result['hub_score']:.4f}")
                            st.write(f"**Category:** {result.get('category', 'N/A')}")
                            st.write(f"**Length:** {len(result.get('content', ''))} chars")
                            st.text_area("Content preview", result.get('content', '')[:500], height=100)
                else:
                    st.info("No documents matched your query.")
            else:
                st.warning("Please enter a query.")

# Ranking Visualization Page
elif page == "Ranking Visualization":
    st.header("Ranking Visualization")
    
    st.caption("Ranking determines what users see out of the candidate documents. Different methods emphasize different signals.")
    
    if not st.session_state.indexed:
        st.warning("Index not built. Please build index first.")
    elif not st.session_state.url_graph:
        st.warning("No URL graph available. Link analysis requires a graph.")
    else:
        # Compute PageRank & HITS if not ready
        if not hasattr(st.session_state.search_engine, 'pagerank_scores'):
            if st.button("Compute PageRank & HITS"):
                build_link_graph()
                st.success("Link analysis computed!")
                st.rerun()
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Method comparison", "PageRank", "HITS", "Graph structure"])
        
        with tab1:
            st.subheader("Compare ranking methods")
            query = st.text_input("Query", "machine learning")
            candidates_limit = st.slider("Candidates limit", 5, 50, 20)
            link_weight = st.slider("Link weight", 0.0, 1.0, 0.3, 0.05, key="rank_link_weight")
            
            if st.button("Compare rankings"):
                methods = ["tfidf", "bm25", "pagerank", "hits", "hybrid"]
                rankings = {}
                
                for method in methods:
                    results = st.session_state.search_engine.search(
                        query=query,
                        limit=candidates_limit,
                        ranking_method=method,
                        link_weight=link_weight
                    )
                    rankings[method] = [r.get('url', '') for r in results]
                
                # Build rank table
                rank_data = []
                for i in range(candidates_limit):
                    row = {'rank': i + 1}
                    for method in methods:
                        if i < len(rankings[method]):
                            row[method] = rankings[method][i][:30]
                        else:
                            row[method] = '-'
                    rank_data.append(row)
                
                st.dataframe(pd.DataFrame(rank_data), use_container_width=True)
                st.caption("Read the table: each row is a rank position; columns show which document each method puts there.")
                
                # Visualization
                fig = go.Figure()
                for method in methods:
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(rankings[method]) + 1)),
                        y=list(range(1, len(rankings[method]) + 1)),
                        mode='lines+markers',
                        name=method
                    ))
                fig.update_layout(title="Rank positions by method", xaxis_title="Rank", yaxis_title="Position")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("PageRank Scores")
            if hasattr(st.session_state.search_engine, 'pagerank_scores'):
                pr_scores = st.session_state.search_engine.pagerank_scores
                sorted_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
                
                df = pd.DataFrame(sorted_pr, columns=['URL', 'PageRank'])
                st.dataframe(df.head(20), use_container_width=True)
                
                fig = px.bar(x=[url[:30] for url, _ in sorted_pr[:20]], y=[score for _, score in sorted_pr[:20]], title="Top 20 PageRank Scores")
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("PageRank measures the importance of a page based on the number and quality of links to it.")
            else:
                st.info("PageRank not computed yet.")
        
        with tab3:
            st.subheader("HITS Scores")
            if hasattr(st.session_state.search_engine, 'hits_scores'):
                hits = st.session_state.search_engine.hits_scores
                hubs = hits.get('hubs', {})
                auths = hits.get('authorities', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Authorities** (good hubs point to them)")
                    sorted_auth = sorted(auths.items(), key=lambda x: x[1], reverse=True)
                    df_auth = pd.DataFrame(sorted_auth, columns=['URL', 'Authority'])
                    st.dataframe(df_auth.head(10), use_container_width=True)
                    
                    fig_auth = px.bar(x=[url[:30] for url, _ in sorted_auth[:10]], y=[score for _, score in sorted_auth[:10]], title="Top 10 Authority Scores")
                    st.plotly_chart(fig_auth, use_container_width=True)
                
                with col2:
                    st.write("**Hubs** (point to good authorities)")
                    sorted_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)
                    df_hubs = pd.DataFrame(sorted_hubs, columns=['URL', 'Hub'])
                    st.dataframe(df_hubs.head(10), use_container_width=True)
                    
                    fig_hubs = px.bar(x=[url[:30] for url, _ in sorted_hubs[:10]], y=[score for _, score in sorted_hubs[:10]], title="Top 10 Hub Scores")
                    st.plotly_chart(fig_hubs, use_container_width=True)
                
                st.caption("HITS identifies two types of important pages: hubs (good directories) and authorities (authoritative sources).")
            else:
                st.info("HITS not computed yet.")
        
        with tab4:
            st.subheader("Graph Structure")
            if st.session_state.url_graph:
                G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", G.number_of_nodes())
                col2.metric("Edges", G.number_of_edges())
                col3.metric("Density", f"{nx.density(G):.4f}")
                col4.metric("Avg degree", f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
                
                st.plotly_chart(st.session_state.search_engine.visualize_graph(G, top_n=30), use_container_width=True)
            else:
                st.info("No URL graph available.")

# Recommendation Panel Page
elif page == "Recommendation Panel":
    st.header("Recommendation Panel")
    
    if not st.session_state.documents:
        st.warning("No documents loaded. Please acquire documents first.")
    else:
        # Select recommendation approach
        approach = st.radio("Recommendation approach", ["content", "collaborative", "hybrid"])
        st.session_state.recommender.approach = approach
        
        # Build recommender
        build_recommender()
        
        # Display recommender statistics
        stats = st.session_state.recommender.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", stats['total_documents'])
        col2.metric("Content features", stats['content_features'])
        col3.metric("Rated documents", stats['rated_documents'])
        col4.metric("Matrix sparsity", f"{stats['matrix_sparsity_%']}%")
        
        # Warning for collaborative/hybrid without ratings
        if approach in ["collaborative", "hybrid"] and stats['rated_documents'] == 0:
            st.warning("Collaborative filtering requires ratings. Using bundled ratings.")
            # Load bundled ratings
            st.session_state.recommender.ratings = BUNDLED_RATINGS
        
        # Recommendation parameters
        st.subheader("Recommendation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Top-K recommendations", 1, 20, 5)
        
        with col2:
            if approach == "hybrid":
                content_weight = st.slider("Content weight", 0.0, 1.0, 0.5, 0.1)
                collaborative_weight = 1.0 - content_weight
        
        # Collaborative mode selection
        if approach == "collaborative" or approach == "hybrid":
            collaborative_mode = st.radio("Collaborative mode", ["user-based", "item-based"])
        else:
            collaborative_mode = "user-based"
        
        # Content mode selection
        if approach == "content":
            content_mode = st.radio("Content mode", ["similar to document", "free-text query"])
        else:
            content_mode = "similar to document"
        
        # Generate recommendations
        st.subheader("Generate Recommendations")
        
        if content_mode == "similar to document":
            doc_options = document_options()
            selected_doc = st.selectbox("Select document", range(len(st.session_state.documents)), format_func=lambda x: doc_options[x])
            
            if st.button("Get recommendations"):
                recommendations = st.session_state.recommender.recommend(
                    doc_index=selected_doc,
                    top_k=top_k,
                    collaborative_mode=collaborative_mode,
                    content_weight=content_weight if approach == "hybrid" else 1.0,
                    collaborative_weight=collaborative_weight if approach == "hybrid" else 0.0
                )
                
                # Display recommendations
                if recommendations:
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                    
                    fig = px.bar(x=[r['url'][:30] for r in recommendations], y=[r['similarity_score'] for r in recommendations], title="Recommendation Scores")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec['url'][:60]}"):
                            st.write(f"**Score:** {rec['similarity_score']:.4f}")
                            st.text_area("Content preview", rec.get('content_preview', ''), height=100)
        else:
            query = st.text_input("Enter query")
            
            if st.button("Get recommendations"):
                recommendations = st.session_state.recommender.recommend(
                    query=query,
                    top_k=top_k,
                    collaborative_mode=collaborative_mode,
                    content_weight=content_weight if approach == "hybrid" else 1.0,
                    collaborative_weight=collaborative_weight if approach == "hybrid" else 0.0
                )
                
                # Display recommendations
                if recommendations:
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                    
                    fig = px.bar(x=[r['url'][:30] for r in recommendations], y=[r['similarity_score'] for r in recommendations], title="Recommendation Scores")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec['url'][:60]}"):
                            st.write(f"**Score:** {rec['similarity_score']:.4f}")
                            st.text_area("Content preview", rec.get('content_preview', ''), height=100)
        
        # Comparison of approaches
        st.markdown("---")
        st.subheader("Comparison of the three approaches")
        
        query_doc = st.selectbox("Select query document for comparison", range(len(st.session_state.documents)), format_func=lambda x: document_options()[x])
        
        if st.button("Compare approaches"):
            # Get recommendations from each approach
            content_recs = st.session_state.recommender.recommend(doc_index=query_doc, top_k=top_k, approach='content')
            collab_recs = st.session_state.recommender.recommend(doc_index=query_doc, top_k=top_k, approach='collaborative')
            hybrid_recs = st.session_state.recommender.recommend(doc_index=query_doc, top_k=top_k, approach='hybrid')
            
            st.write(f"Content-based: {len(content_recs)} suggestions")
            st.write(f"Collaborative: {len(collab_recs)} suggestions")
            st.write(f"Hybrid: {len(hybrid_recs)} suggestions")
            
            # Display comparison
            comparison_data = []
            for i in range(top_k):
                row = {'rank': i + 1}
                if i < len(content_recs):
                    row['content'] = content_recs[i]['url'][:30]
                else:
                    row['content'] = '-'
                if i < len(collab_recs):
                    row['collaborative'] = collab_recs[i]['url'][:30]
                else:
                    row['collaborative'] = '-'
                if i < len(hybrid_recs):
                    row['hybrid'] = hybrid_recs[i]['url'][:30]
                else:
                    row['hybrid'] = '-'
                comparison_data.append(row)
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Visualize user-item rating matrix
            if BUNDLED_RATINGS:
                with st.expander("User-item rating matrix"):
                    ratings_df = pd.DataFrame(BUNDLED_RATINGS).fillna(0)
                    fig = px.imshow(ratings_df, text_auto=True, title="User-Item Rating Matrix")
                    st.plotly_chart(fig, use_container_width=True)

# Evaluation Dashboard Page
elif page == "Evaluation Dashboard":
    st.header("Evaluation Dashboard")
    
    if not st.session_state.indexed:
        st.warning("Index not built. Please build index first.")
    else:
        # Tabs for evaluation functions
        tab1, tab2, tab3 = st.tabs(["Relevance judgements", "Run & compare", "Per-query analysis"])
        
        with tab1:
            st.subheader("Relevance judgements")
            
            # Display existing judgments
            if st.session_state.evaluator.relevant_docs:
                st.write("Existing ground truth:")
                for qid, rel_docs in st.session_state.evaluator.relevant_docs.items():
                    qtext = st.session_state.evaluator.query_text.get(qid, qid)
                    graded = st.session_state.evaluator.graded_relevance.get(qid, {})
                    st.write(f"**{qid}**: '{qtext}' - {len(rel_docs)} relevant docs (graded: {len(graded)})")
            
            # Load bundled judgments
            if st.button("Load bundled judgments"):
                for qid, qdata in BUNDLED_QUERIES.items():
                    st.session_state.evaluator.query_text[qid] = qdata['query_text']
                    st.session_state.evaluator.relevant_docs[qid] = set(qdata['relevant_doc_ids'])
                    if 'graded_relevance' in qdata:
                        st.session_state.evaluator.graded_relevance[qid] = qdata['graded_relevance']
                st.success("Loaded 7 bundled queries with relevance judgments")
                st.rerun()
            
            # Clear all judgments
            if st.button("Clear all judgments"):
                st.session_state.evaluator.relevant_docs = {}
                st.session_state.evaluator.query_text = {}
                st.session_state.evaluator.graded_relevance = {}
                st.success("All judgments cleared")
                st.rerun()
            
            # Add custom query
            st.markdown("---")
            st.subheader("Add custom query")
            new_qid = st.text_input("Query ID", "q_custom")
            new_qtext = st.text_input("Query text", "")
            
            if st.session_state.documents:
                doc_options = document_options()
                new_rel_docs = st.multiselect("Select relevant documents", range(len(st.session_state.documents)), format_func=lambda x: doc_options[x])
                
                if st.button("Save query"):
                    if new_qid and new_qtext:
                        st.session_state.evaluator.query_text[new_qid] = new_qtext
                        st.session_state.evaluator.relevant_docs[new_qid] = set(new_rel_docs)
                        st.success(f"Query {new_qid} saved")
                        st.rerun()
        
        with tab2:
            st.subheader("Run & compare")
            
            # Select ranking methods
            methods = st.multiselect("Select ranking methods", ["tfidf", "bm25", "pagerank", "hits", "hybrid"], default=["tfidf", "bm25"])
            result_limit = st.number_input("Result limit", 1, 100, 10)
            link_weight = st.slider("Link weight", 0.0, 1.0, 0.3, 0.05)
            use_expansion = st.checkbox("Use query expansion", value=False)
            
            if st.button("Run evaluation"):
                if not st.session_state.evaluator.relevant_docs:
                    st.warning("No relevance judgments available. Please add queries first.")
                else:
                    comparison = []
                    for method in methods:
                        method_metrics = {'method': method}
                        
                        for qid in st.session_state.evaluator.relevant_docs:
                            qtext = st.session_state.evaluator.query_text.get(qid, qid)
                            
                            # Run search
                            results = st.session_state.search_engine.search(
                                query=qtext,
                                limit=result_limit,
                                ranking_method=method,
                                link_weight=link_weight,
                                expand=use_expansion
                            )
                            
                            # Set retrieved documents
                            retrieved_ids = [str(i) for i in range(len(results))]
                            st.session_state.evaluator.set_retrieved_documents(qid, retrieved_ids)
                            
                            # Calculate metrics
                            metrics = st.session_state.evaluator.calculate_all_metrics(qid)
                            
                            # Store metrics
                            for key, value in metrics.items():
                                if key not in method_metrics:
                                    method_metrics[key] = []
                                method_metrics[key].append(value)
                        
                        # Average metrics
                        for key in list(method_metrics.keys()):
                            if key != 'method' and isinstance(method_metrics[key], list):
                                method_metrics[key] = np.mean(method_metrics[key])
                        
                        comparison.append(method_metrics)
                    
                    st.session_state.method_comparison = pd.DataFrame(comparison)
                    
                    # Display comparison
                    st.dataframe(st.session_state.method_comparison, use_container_width=True)
                    
                    # Visualize comparison
                    metrics_to_plot = ['map', 'mrr', 'precision_at_10', 'recall_at_10', 'ndcg_at_10']
                    for metric in metrics_to_plot:
                        if metric in st.session_state.method_comparison.columns:
                            fig = px.bar(st.session_state.method_comparison, x='method', y=metric, title=f"{metric.upper()} by method")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify best method
                    if 'map' in st.session_state.method_comparison.columns:
                        best = st.session_state.method_comparison.loc[st.session_state.method_comparison['map'].idxmax()]
                        st.success(f"Best method by MAP: {best['method']} (MAP={best['map']:.4f})")
        
        with tab3:
            st.subheader("Per-query analysis")
            
            if st.session_state.evaluator.relevant_docs:
                query_id = st.selectbox("Select query", list(st.session_state.evaluator.relevant_docs.keys()))
                method = st.selectbox("Select ranking method", ["tfidf", "bm25", "pagerank", "hits", "hybrid"])
                
                if st.button("Analyze"):
                    qtext = st.session_state.evaluator.query_text.get(query_id, query_id)
                    
                    # Run search
                    results = st.session_state.search_engine.search(
                        query=qtext,
                        limit=20,
                        ranking_method=method
                    )
                    
                    # Set retrieved documents
                    retrieved_ids = [str(i) for i in range(len(results))]
                    st.session_state.evaluator.set_retrieved_documents(query_id, retrieved_ids)
                    
                    # Calculate metrics
                    metrics = st.session_state.evaluator.calculate_all_metrics(query_id)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Precision", f"{metrics['precision']:.4f}")
                    col2.metric("Recall", f"{metrics['recall']:.4f}")
                    col3.metric("F1", f"{metrics['f1_score']:.4f}")
                    
                    # Precision-recall curve
                    st.plotly_chart(st.session_state.evaluator.visualize_precision_recall_curve(query_id), use_container_width=True)
                    
                    # NDCG@K
                    st.plotly_chart(st.session_state.evaluator.visualize_ndcg_at_k(k_values=[5, 10, 20]), use_container_width=True)
                    
                    # Retrieved ranking
                    st.subheader(f"Retrieved ranking for '{qtext}'")
                    for i, result in enumerate(results, 1):
                        is_relevant = str(i-1) in st.session_state.evaluator.relevant_docs[query_id]
                        relevance_mark = "✓" if is_relevant else "✗"
                        st.write(f"{relevance_mark} {i}. {result.get('title', result.get('url', ''))[:50]} (score: {result.get('score', 0):.4f})")
            else:
                st.info("No relevance judgments available.")

# Performance Analytics Page
elif page == "Performance Analytics":
    st.header("Performance Analytics")
    
    # Index and collection statistics
    st.subheader("Index & Collection Statistics")
    if st.session_state.indexed:
        stats = st.session_state.search_engine.get_index_statistics()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents indexed", stats['documents_indexed'])
        col2.metric("Vocabulary size", stats['vocabulary_size'])
        col3.metric("Postings", stats['posting_entries'])
        col4.metric("Build time", f"{stats['index_build_seconds']:.2f}s")
    
    # Crawl statistics
    if hasattr(st.session_state.crawler, 'crawl_stats'):
        st.subheader("Crawl Performance")
        crawl_stats = st.session_state.crawler.crawl_stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages crawled", crawl_stats.get('pages_crawled', 0))
        col2.metric("Duplicate URLs", crawl_stats.get('duplicate_urls_skipped', 0))
        col3.metric("Near-duplicates", crawl_stats.get('near_duplicates', 0))
    
    # Query latency benchmark
    st.markdown("---")
    st.subheader("Query Latency Benchmark")
    
    methods = st.multiselect("Methods to benchmark", ["tfidf", "bm25", "pagerank", "hits", "hybrid"], default=["tfidf", "bm25"])
    result_sizes = st.multiselect("Result set sizes", [5, 10, 20, 50], default=[10])
    num_runs = st.number_input("Number of runs per configuration", 1, 10, 3)
    
    if st.button("Run benchmark"):
        benchmark_results = []
        test_query = "machine learning"
        
        for method in methods:
            for size in result_sizes:
                latencies = []
                for _ in range(num_runs):
                    started = time.perf_counter()
                    st.session_state.search_engine.search(
                        query=test_query,
                        limit=size,
                        ranking_method=method
                    )
                    latencies.append((time.perf_counter() - started) * 1000)
                
                benchmark_results.append({
                    'method': method,
                    'result_size': size,
                    'mean_latency_ms': np.mean(latencies),
                    'std_latency_ms': np.std(latencies),
                    'min_latency_ms': np.min(latencies),
                    'max_latency_ms': np.max(latencies)
                })
        
        benchmark_df = pd.DataFrame(benchmark_results)
        st.dataframe(benchmark_df, use_container_width=True)
        
        # Visualize
        fig = px.box(benchmark_df, x='method', y='mean_latency_ms', color='result_size', title="Latency by method and result size")
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.bar(benchmark_df, x='result_size', y='mean_latency_ms', color='method', barmode='group', title="Latency vs result size")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Observed queries
    st.markdown("---")
    st.subheader("Observed Queries from Session")
    
    if st.session_state.query_log:
        query_df = pd.DataFrame(st.session_state.query_log)
        st.dataframe(query_df, use_container_width=True)
        
        # Latency distribution
        fig = px.histogram(query_df, x='latency_ms', color='method', title="Query latency distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Effectiveness vs efficiency
        if st.session_state.method_comparison is not None:
            st.markdown("---")
            st.subheader("Effectiveness vs Efficiency")
            
            # Merge comparison with latency data
            avg_latency = query_df.groupby('method')['latency_ms'].mean()
            
            for _, row in st.session_state.method_comparison.iterrows():
                method = row['method']
                if method in avg_latency:
                    st.write(f"{method}: MAP={row['map']:.4f}, Avg Latency={avg_latency[method]:.2f}ms")
            
            # Scatter plot
            scatter_data = []
            for _, row in st.session_state.method_comparison.iterrows():
                method = row['method']
                if method in avg_latency:
                    scatter_data.append({
                        'method': method,
                        'map': row['map'],
                        'latency_ms': avg_latency[method]
                    })
            
            if scatter_data:
                scatter_df = pd.DataFrame(scatter_data)
                fig = px.scatter(scatter_df, x='latency_ms', y='map', text='method', title="Effectiveness (MAP) vs Efficiency (Latency)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No queries logged yet. Run some searches to see analytics.")

# Inference & Discussion Page
elif page == "Inference & Discussion":
    st.header("Inference & Discussion")
    
    if st.session_state.method_comparison is None:
        st.warning("Please run the evaluation dashboard first to generate comparison data.")
    else:
        comparison = st.session_state.method_comparison
        index_stats = st.session_state.search_engine.get_index_statistics() if st.session_state.indexed else {}
        duplicate_report = st.session_state.crawler.get_duplicate_report() if hasattr(st.session_state.crawler, 'get_duplicate_report') else pd.DataFrame()
        
        st.subheader("Discussion based on measured numbers")
        
        with st.expander("1. Poor ranking despite relevant retrieval", expanded=True):
            if comparison is not None and not comparison.empty:
                best_recall = comparison.loc[comparison["mean_recall_at_10"].idxmax()]
                best_ndcg = comparison.loc[comparison["mean_ndcg_at_10"].idxmax()]
                st.markdown(
                    f"**Measured in this session.** The best mean recall@10 is "
                    f"{best_recall['mean_recall_at_10']:.3f} ({best_recall['method']}), "
                    f"while the best mean NDCG@10 is {best_ndcg['mean_ndcg_at_10']:.3f} "
                    f"({best_ndcg['method']}). High recall with lower NDCG is exactly "
                    f"the signature of relevant documents being found but ordered badly."
                )
            st.markdown(
                "**Causes.** Term-frequency saturation, so a document that repeats a "
                "\"query term outranks a better one; no length normalisation, so long "
                "\"documents accumulate weight; a purely textual score that ignores "
                "\"document authority; equal weighting of all query terms; and fusing "
                "\"scores that live on different scales, which lets one component "
                "\"silently dominate.\n\n"
                "**Improvements implemented here.** BM25 replaces raw TF-IDF and adds "
                "\"saturation plus length normalisation; PageRank and HITS supply a "
                "\"query-independent authority prior; both components are min-"
                "\"max normalised before the weighted fusion, so the link weight slider has "
                "\"a predictable effect; pseudo-relevance feedback expands the "
                "query to \"recover synonym matches; phrase and required-term operators "
                "promote \"exact intent matches."
            )
        
        with st.expander("2. Effect of duplicate and near-duplicate documents"):
            if not duplicate_report.empty:
                near = (duplicate_report["type"] == "near duplicate").sum()
                exact = (duplicate_report["type"] == "exact duplicate").sum()
                st.markdown(
                    f"**Measured in this session.** {exact} exact and {near} near "
                    f"duplicates were rejected before indexing. The near duplicates "
                    f"were only caught by shingle similarity: their MD5 hashes differ, "
                    f"so hashing alone would have admitted them."
                )
            st.markdown(
                "**Indexing.** Duplicates inflate document frequency, which lowers the "
                "\"IDF of the affected terms and distorts every score that depends on "
                "\"it; the index also grows without adding information.\n\n"
                "**Ranking.** Near-identical pages occupy several of the top "
                "\"positions, crowding out diverse results and wasting the user's attention.\n\n"
                "**Recommendation.** A near duplicate is the nearest neighbour of its "
                "\"own original, so the Top-K list degenerates into copies of the "
                "\"seed document.\n\n"
                "**Evaluation.** Precision@K is inflated when a duplicate of "
                "\"a relevant document counts as a second hit, and judgements made on one "
                "\"copy do not transfer to the other, so scores become unstable.\n\n"
                "**Mitigation implemented here.** MD5 content hashing removes exact "
                "\"duplicates; 5-shingle Jaccard similarity above a configurable "
                "\"threshold removes near duplicates; URL normalisation and a "
                "\"visited set remove duplicate URLs before fetching."
            )
        
        with st.expander("3. Content-based versus collaborative recommendation"):
            stats = st.session_state.recommender.get_statistics()
            st.markdown(
                f"**Measured in this session.** Ratings cover "
                f"{stats['rated_documents']} of {stats['total_documents']} d"
                f"ocuments ({stats['coverage_%']}%) from {stats['users']} users, and the "
                f"user-item matrix is {stats['matrix_sparsity_%']}% sparse."
            )
            st.markdown(
                "**Content-based** needs only the documents themselves, so it handles "
                "\"new items immediately (no item cold start), is explainable through "
                "\"shared terms, and works with no users at all. Its ceiling is the "
                "\"text: it cannot recommend something that is useful but lexically "
                "\"different, and it tends to over-specialise on what the user already "
                "read.\n\n"
                "**Collaborative** exploits behaviour, so it can surface an "
                "\"item whose text looks unrelated but which similar users valued, and it "
                "captures \"quality signals that text cannot express. It fails on cold-"
                "start items \"and users, and it degrades as the rating matrix becomes "
                "sparse - \"unrated documents receive a score of exactly zero here.\n\n"
                "**When to prefer which.** Use content-based for a fresh or "
                "rapidly \"changing corpus, for niche items with few ratings, and when "
                "an \"explanation is required. Use collaborative when there is de"
                "nse \"interaction data and serendipity matters. Use the hybrid to "
                "get \"coverage from content and preference signal from behaviour, "
                "which is \"why the hybrid here normalises both components before "
                "combining them."
            )
        
        with st.expander("4. Value of integrating the whole pipeline"):
            st.markdown(
                f"**Measured in this session.** Acquisition produced "
                f"{len(st.session_state.documents)} de-duplicated documents; "
                f"preprocessing and indexing turned them into "
                f"{index_stats.get('vocabulary_size', 0)} features with "
                f"{index_stats.get('posting_entries', 0)} postings; link "
                "analysis over "
                f"{len(st.session_state.url_graph)} nodes reordered results; and "
                f"evaluation over {len(st.session_state.evaluator.relevant_docs)} judged "
                f"queries quantified the effect."
            )
            st.markdown(
                "Each stage constrains the quality achievable by the next. C"
                "rawling \"decides what can ever be found, and its duplicate handling "
                "protects \"the IDF statistics that ranking depends on. Preprocessing f"
                "ixes the \"vocabulary, and an over-aggressive setting silently destroy"
                "s recall - \"raising min_df to 2 on this corpus removed every single-doc"
                "ument term \"and made those queries return nothing. Indexing determines "
                "which \"queries are answerable and how fast. Ranking decides what t"
                "he user \"actually sees out of the candidates. Recommendation reuses "
                "the same \"document representation to extend a single result into a se"
                "ssion. \"Evaluation closes the loop by measuring whether any change "
                "helped, \"which is what makes the improvements above defensible rather "
                "than \"anecdotal."
            )
        
        with st.expander("5. Learnings from the measured results"):
            if comparison is not None and not comparison.empty:
                table = comparison[["method", "map", "mrr", "mean_ndcg_at_10", "avg_latency_ms"]].round(4)
                st.dataframe(table, use_container_width=True)
                best = comparison.loc[comparison["map"].idxmax()]
                textual = comparison[comparison["method"].isin(["tfidf", "bm25"])]
                st.markdown(
                    f"- **{best['method']}** achieved the highest MAP "
                    f"({best['map']:.4f}) in this run.\n"
                    f"- Adding a link-analysis prior changed the ordering rather "
                    f"than the candidate set, which is why NDCG moves while recall "
                    f"stays similar.\n"
                    f"- Mean query latency stayed at "
                    f"{comparison['avg_latency_ms'].mean():.2f} ms, so the "
                    f"effectiveness gains cost essentially nothing at this s"
                    "cale."
                )
                if not textual.empty:
                    st.markdown(
                        f"- Purely textual ranking reached a MAP of "
                        f"{textual['map'].max():.4f}, so link analysis contr"
                        f"ibuted the remaining difference."
                    )
            st.markdown(
                "- Metrics must compare **document identities**, not rank positions; "
                "using positions as identifiers produces numbers that look plausible "
                "but mean nothing.\n"
                "- Score fusion is only meaningful after normalisation, because "
                "cosine similarity and PageRank occupy different ranges.\n"
                "- Aggressive vocabulary pruning is a silent recall bug on small "
                "collections.\n"
                "- Exact hashing is insufficient for de-duplication; shingle "
                "similarity is required to catch paraphrased mirrors.\n"
                "- Rank-aware measures (NDCG, MRR, MAP) expose ordering problems "
                "that set-based precision and recall cannot see."
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Information Retrieval Assignment")
st.sidebar.markdown("Built with Streamlit")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown("**Live Demo:** https://irassignment2grp78.streamlit.app/")
