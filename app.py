import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import networkx as nx

# Import our modules
from web_crawler import WebCrawler
from text_preprocessing import TextMiningFramework
from search_engine import SearchEngine
from recommender_system import RecommenderSystem
from evaluation_metrics import IREvaluation

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #2ecc71;
        font-weight: bold;
    }
    .warning-message {
        color: #f39c12;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Dashboard", "Web Crawling", "Text Preprocessing", "Index Management", 
     "Web Search", "Ranking Visualization", "Recommendation Panel", 
     "Evaluation Dashboard", "Performance Analytics"]
)

# Main header
st.markdown('<h1 class="main-header">Information Retrieval System</h1>', unsafe_allow_html=True)

# Dashboard Page
if page == "Dashboard":
    st.header("System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Crawled", len(st.session_state.documents))
    with col2:
        st.metric("Documents Indexed", "Yes" if st.session_state.indexed else "No")
    with col3:
        st.metric("Documents Processed", "Yes" if st.session_state.processed else "No")
    with col4:
        st.metric("URLs in Graph", len(st.session_state.url_graph))
    
    st.subheader("System Status")
    
    if st.session_state.documents:
        st.success(f"✓ {len(st.session_state.documents)} documents loaded")
    else:
        st.warning("⚠ No documents loaded - Start with Web Crawling")
    
    if st.session_state.indexed:
        st.success("✓ Search index created")
    else:
        st.warning("⚠ No search index - Process documents first")
    
    if st.session_state.processed:
        st.success("✓ Text preprocessing completed")
    else:
        st.warning("⚠ Text preprocessing not completed")
    
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load Sample Data"):
            # Create sample documents
            sample_docs = [
                {
                    'url': 'https://example.com/doc1',
                    'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning techniques.'
                },
                {
                    'url': 'https://example.com/doc2',
                    'content': 'Deep learning uses neural networks with multiple layers to model complex patterns in data. It has achieved remarkable success in image recognition, natural language processing, and speech recognition.'
                },
                {
                    'url': 'https://example.com/doc3',
                    'content': 'Natural language processing enables computers to understand and generate human language. Applications include machine translation, sentiment analysis, and chatbots.'
                },
                {
                    'url': 'https://example.com/doc4',
                    'content': 'Computer vision allows machines to interpret and understand visual information from the world. Key tasks include object detection, image classification, and facial recognition.'
                },
                {
                    'url': 'https://example.com/doc5',
                    'content': 'Reinforcement learning trains agents to make decisions by rewarding desired behaviors. It has been successfully applied to game playing, robotics, and autonomous systems.'
                }
            ]
            sample_metadata = [
                {'title': 'Introduction to Machine Learning', 'url': 'https://example.com/doc1'},
                {'title': 'Deep Learning Overview', 'url': 'https://example.com/doc2'},
                {'title': 'Natural Language Processing', 'url': 'https://example.com/doc3'},
                {'title': 'Computer Vision Applications', 'url': 'https://example.com/doc4'},
                {'title': 'Reinforcement Learning', 'url': 'https://example.com/doc5'}
            ]
            sample_graph = {
                'https://example.com/doc1': ['https://example.com/doc2', 'https://example.com/doc3'],
                'https://example.com/doc2': ['https://example.com/doc4'],
                'https://example.com/doc3': ['https://example.com/doc5'],
                'https://example.com/doc4': ['https://example.com/doc5'],
                'https://example.com/doc5': []
            }
            
            st.session_state.documents = sample_docs
            st.session_state.metadata = sample_metadata
            st.session_state.url_graph = sample_graph
            st.success("Sample data loaded successfully!")
            st.rerun()
    
    with col2:
        if st.button("Process Documents"):
            if st.session_state.documents:
                st.session_state.text_mining.load_documents(st.session_state.documents)
                st.session_state.text_mining.extract_tfidf_features()
                st.session_state.processed = True
                st.success("Documents processed successfully!")
                st.rerun()
            else:
                st.error("No documents to process!")
    
    with col3:
        if st.button("Create Index"):
            if st.session_state.documents and st.session_state.processed:
                st.session_state.search_engine.create_index(
                    st.session_state.documents,
                    st.session_state.metadata
                )
                if st.session_state.url_graph:
                    G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
                    st.session_state.search_engine.calculate_pagerank(G)
                st.session_state.indexed = True
                st.success("Index created successfully!")
                st.rerun()
            else:
                st.error("Process documents first!")

# Web Crawling Page
elif page == "Web Crawling":
    st.header("Web Crawling Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Crawl Configuration")
        
        seed_urls_input = st.text_area(
            "Seed URLs (one per line)",
            "https://en.wikipedia.org/wiki/Machine_learning\nhttps://en.wikipedia.org/wiki/Deep_learning",
            height=100
        )
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            max_depth = st.number_input("Max Depth", min_value=1, max_value=5, value=2)
        with col_b:
            max_pages = st.number_input("Max Pages", min_value=1, max_value=500, value=50)
        with col_c:
            stay_on_domain = st.checkbox("Stay on Domain", value=True)
        
        if st.button("Start Crawling"):
            seed_urls = [url.strip() for url in seed_urls_input.split('\n') if url.strip()]
            
            with st.spinner("Crawling in progress..."):
                stats = st.session_state.crawler.crawl(
                    seed_urls=seed_urls,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    stay_on_domain=stay_on_domain
                )
                
                st.session_state.documents = st.session_state.crawler.documents
                st.session_state.metadata = st.session_state.crawler.metadata
                st.session_state.url_graph = st.session_state.crawler.url_graph
                
                # Save data
                st.session_state.crawler.save_documents('documents.json')
                st.session_state.crawler.save_metadata('metadata.csv')
                st.session_state.crawler.save_graph('url_graph.json')
            
            st.success(f"Crawling completed! {stats['pages_crawled']} pages crawled.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Pages Crawled", stats['pages_crawled'])
            col2.metric("Duplicate URLs Skipped", stats['duplicate_urls_skipped'])
            col3.metric("Duplicate Documents Skipped", stats['duplicate_documents_skipped'])
    
    with col2:
        st.subheader("Load Existing Data")
        
        if st.button("Load from Files"):
            if os.path.exists('documents.json'):
                st.session_state.crawler.load_documents('documents.json')
                st.session_state.documents = st.session_state.crawler.documents
                st.success("Documents loaded!")
            
            if os.path.exists('metadata.csv'):
                st.session_state.crawler.load_metadata('metadata.csv')
                st.session_state.metadata = st.session_state.crawler.metadata
                st.success("Metadata loaded!")
            
            if os.path.exists('url_graph.json'):
                st.session_state.crawler.load_graph('url_graph.json')
                st.session_state.url_graph = st.session_state.crawler.url_graph
                st.success("URL graph loaded!")
    
    if st.session_state.documents:
        st.subheader("Crawled Documents")
        doc_df = st.session_state.crawler.get_documents_dataframe()
        st.dataframe(doc_df[['url', 'hash']], use_container_width=True)

# Text Preprocessing Page
elif page == "Text Preprocessing":
    st.header("Text Preprocessing and Mining")
    
    if not st.session_state.documents:
        st.warning("No documents loaded. Please crawl or load documents first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Preprocessing Options")
            
            remove_stops = st.checkbox("Remove Stopwords", value=True)
            lemmatize = st.checkbox("Lemmatize", value=True)
            
            if st.button("Process Documents"):
                st.session_state.text_mining.load_documents(st.session_state.documents)
                st.session_state.text_mining.extract_tfidf_features()
                st.session_state.processed = True
                st.success("Documents processed successfully!")
                st.rerun()
        
        with col2:
            st.subheader("Corpus Statistics")
            if st.session_state.processed:
                stats = st.session_state.text_mining.get_corpus_statistics()
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Total Words", stats['total_words'])
                st.metric("Vocabulary Size", stats['vocabulary_size'])
                st.metric("Avg Doc Length", f"{stats['average_document_length']:.1f}")
        
        if st.session_state.processed:
            st.subheader("Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    st.session_state.text_mining.visualize_document_lengths(),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    st.session_state.text_mining.visualize_vocabulary_distribution(),
                    use_container_width=True
                )
            
            st.subheader("Topic Modeling")
            n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)
            
            if st.button("Run Topic Modeling"):
                topics = st.session_state.text_mining.perform_topic_modeling(n_topics=n_topics)
                
                for i, topic in enumerate(topics):
                    st.write(f"**Topic {i+1}:** {', '.join(topic)}")
                
                st.plotly_chart(
                    st.session_state.text_mining.visualize_topic_distribution(topics),
                    use_container_width=True
                )
            
            st.subheader("Document Clustering")
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            
            if st.button("Cluster Documents"):
                cluster_result = st.session_state.text_mining.cluster_documents(n_clusters=n_clusters)
                
                st.plotly_chart(
                    st.session_state.text_mining.visualize_clusters(cluster_result),
                    use_container_width=True
                )
                
                for cluster_id, terms in cluster_result['cluster_terms'].items():
                    st.write(f"**Cluster {cluster_id}:** {', '.join(terms)}")

# Index Management Page
elif page == "Index Management":
    st.header("Index Management")
    
    if not st.session_state.documents:
        st.warning("No documents loaded. Please crawl or load documents first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Create Search Index")
            
            if st.button("Create/Rebuild Index"):
                st.session_state.search_engine.create_index(
                    st.session_state.documents,
                    st.session_state.metadata
                )
                st.session_state.indexed = True
                st.success("Index created successfully!")
        
        with col2:
            st.subheader("Build URL Graph")
            
            if st.session_state.url_graph:
                if st.button("Build Graph & Calculate PageRank"):
                    G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
                    pr_scores = st.session_state.search_engine.calculate_pagerank(G)
                    st.success(f"Graph built with {len(pr_scores)} nodes")
            else:
                st.info("No URL graph available")
        
        if st.session_state.indexed:
            st.subheader("Index Statistics")
            st.metric("Index Status", "Active")
            st.metric("Documents Indexed", len(st.session_state.documents))
            
            if st.session_state.url_graph:
                st.subheader("Graph Statistics")
                G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
                st.metric("Nodes", G.number_of_nodes())
                st.metric("Edges", G.number_of_edges())

# Web Search Page
elif page == "Web Search":
    st.header("Web Search Interface")
    
    if not st.session_state.indexed:
        st.warning("Search index not created. Please create index first.")
    else:
        st.subheader("Search Query")
        
        query = st.text_input("Enter your search query:", "")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ranking_method = st.selectbox("Ranking Method", ["tfidf", "pagerank", "hits"])
        with col2:
            limit = st.number_input("Number of Results", min_value=1, max_value=50, value=10)
        with col3:
            st.empty()
        
        if st.button("Search"):
            if query:
                results = st.session_state.search_engine.search(
                    query=query,
                    limit=limit,
                    ranking_method=ranking_method
                )
                
                st.subheader(f"Search Results ({len(results)} found)")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['title'] or result['url']}"):
                        st.write(f"**URL:** {result['url']}")
                        st.write(f"**Score:** {result['score']:.4f}")
                        if ranking_method == 'pagerank' and 'pagerank_score' in result:
                            st.write(f"**PageRank Score:** {result['pagerank_score']:.4f}")
                        if ranking_method == 'hits' and 'authority_score' in result:
                            st.write(f"**Authority Score:** {result['authority_score']:.4f}")
                        st.write(f"**Content Preview:** {result['content']}")
            else:
                st.warning("Please enter a search query.")
        
        st.subheader("Advanced Search")
        
        with st.expander("Advanced Options"):
            min_length = st.number_input("Minimum Document Length", min_value=0, value=0)
            must_contain = st.text_input("Must Contain Word", "")
            
            if st.button("Advanced Search"):
                filters = {}
                if min_length > 0:
                    filters['min_length'] = min_length
                if must_contain:
                    filters['must_contain'] = must_contain
                
                results = st.session_state.search_engine.advanced_search(
                    query=query,
                    filters=filters,
                    limit=limit
                )
                
                st.write(f"Found {len(results)} results with filters")

# Ranking Visualization Page
elif page == "Ranking Visualization":
    st.header("Ranking Visualization")
    
    if not st.session_state.url_graph:
        st.warning("No URL graph available. Please crawl documents first.")
    else:
        st.subheader("PageRank Scores")
        
        if st.button("Calculate and Visualize PageRank"):
            G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
            pr_scores = st.session_state.search_engine.calculate_pagerank(G)
            
            st.plotly_chart(
                st.session_state.search_engine.visualize_pagerank_scores(top_n=20),
                use_container_width=True
            )
        
        st.subheader("HITS Scores")
        
        if st.button("Calculate and Visualize HITS"):
            G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
            hubs, authorities = st.session_state.search_engine.calculate_hits(G)
            
            st.plotly_chart(
                st.session_state.search_engine.visualize_hits_scores(top_n=20),
                use_container_width=True
            )
        
        st.subheader("URL Graph Structure")
        
        if st.button("Visualize Graph"):
            G = st.session_state.search_engine.build_graph(st.session_state.url_graph)
            st.plotly_chart(
                st.session_state.search_engine.visualize_graph(G, top_n=30),
                use_container_width=True
            )

# Recommendation Panel Page
elif page == "Recommendation Panel":
    st.header("Recommendation Panel")
    
    if not st.session_state.processed:
        st.warning("Documents not processed. Please process documents first.")
    else:
        st.subheader("Recommendation Settings")
        
        approach = st.selectbox("Recommendation Approach", ["content", "hybrid"])
        
        if approach == "content":
            st.session_state.recommender.fit(
                st.session_state.documents,
                approach='content'
            )
        elif approach == "hybrid":
            st.session_state.recommender.fit(
                st.session_state.documents,
                approach='hybrid'
            )
        
        st.subheader("Get Recommendations")
        
        rec_method = st.radio("Method", ["By Document", "By Query"])
        
        if rec_method == "By Document":
            doc_options = [f"{i}: {doc['url'][:50]}" for i, doc in enumerate(st.session_state.documents)]
            selected_doc = st.selectbox("Select Document", range(len(st.session_state.documents)), 
                                      format_func=lambda x: doc_options[x])
            
            top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
            
            if st.button("Get Recommendations"):
                recommendations = st.session_state.recommender.recommend(
                    doc_index=selected_doc,
                    top_k=top_k
                )
                
                st.subheader(f"Top {len(recommendations)} Recommendations")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['url'][:60]}"):
                        st.write(f"**Similarity Score:** {rec['similarity_score']:.4f}")
                        st.write(f"**Content Preview:** {rec['content_preview']}")
                
                if recommendations:
                    st.plotly_chart(
                        st.session_state.recommender.visualize_recommendations(recommendations),
                        use_container_width=True
                    )
        
        else:
            query = st.text_input("Enter Query:", "")
            top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
            
            if st.button("Get Recommendations"):
                recommendations = st.session_state.recommender.recommend(
                    query=query,
                    top_k=top_k
                )
                
                st.subheader(f"Top {len(recommendations)} Recommendations")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['url'][:60]}"):
                        st.write(f"**Similarity Score:** {rec['similarity_score']:.4f}")
                        st.write(f"**Content Preview:** {rec['content_preview']}")
                
                if recommendations:
                    st.plotly_chart(
                        st.session_state.recommender.visualize_recommendations(recommendations),
                        use_container_width=True
                    )

# Evaluation Dashboard Page
elif page == "Evaluation Dashboard":
    st.header("Evaluation Dashboard")
    
    st.subheader("Setup Evaluation")
    
    st.write("To evaluate the system, you need to define relevant documents for queries.")
    
    # Query setup
    query_id = st.text_input("Query ID", "q1")
    query_text = st.text_input("Query Text", "machine learning")
    
    # Select relevant documents
    if st.session_state.documents:
        doc_options = [f"{i}: {doc['url'][:50]}" for i, doc in enumerate(st.session_state.documents)]
        relevant_docs = st.multiselect(
            "Select Relevant Documents",
            range(len(st.session_state.documents)),
            format_func=lambda x: doc_options[x]
        )
        
        if st.button("Add Query to Evaluation"):
            st.session_state.evaluator.set_relevant_documents(
                query_id,
                set([str(i) for i in relevant_docs])
            )
            st.success(f"Query {query_id} added to evaluation!")
    
    # Run search for evaluation
    st.subheader("Run Search for Evaluation")
    
    eval_query_id = st.selectbox("Select Query for Evaluation", 
                                 list(st.session_state.evaluator.relevant_docs.keys()))
    
    if eval_query_id:
        ranking_method = st.selectbox("Ranking Method", ["tfidf", "pagerank", "hits"])
        
        if st.button("Run Search & Evaluate"):
            # Run search
            results = st.session_state.search_engine.search(
                query=eval_query_id.replace('q', ''),  # Simple extraction
                limit=20,
                ranking_method=ranking_method
            )
            
            # Get retrieved document IDs
            retrieved_ids = [str(i) for i in range(len(results))]
            st.session_state.evaluator.set_retrieved_documents(eval_query_id, retrieved_ids)
            
            # Calculate metrics
            metrics = st.session_state.evaluator.calculate_all_metrics(eval_query_id)
            
            st.subheader("Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precision", f"{metrics['precision']:.4f}")
                st.metric("Recall", f"{metrics['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            with col2:
                st.metric("Average Precision", f"{metrics['average_precision']:.4f}")
                st.metric("Reciprocal Rank", f"{metrics['reciprocal_rank']:.4f}")
            
            with col3:
                st.metric("Precision@10", f"{metrics['precision_at_10']:.4f}")
                st.metric("Recall@10", f"{metrics['recall_at_10']:.4f}")
                st.metric("NDCG@10", f"{metrics['ndcg_at_10']:.4f}")
            
            # Precision-Recall Curve
            st.subheader("Precision-Recall Curve")
            st.plotly_chart(
                st.session_state.evaluator.visualize_precision_recall_curve(eval_query_id),
                use_container_width=True
            )

# Performance Analytics Page
elif page == "Performance Analytics":
    st.header("Performance Analytics")
    
    st.subheader("System-wide Metrics")
    
    if st.session_state.evaluator.relevant_docs:
        system_metrics = st.session_state.evaluator.calculate_system_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAP", f"{system_metrics['map']:.4f}")
            st.metric("MRR", f"{system_metrics['mrr']:.4f}")
        
        with col2:
            st.metric("Mean Precision@10", f"{system_metrics['mean_precision_at_10']:.4f}")
            st.metric("Mean Recall@10", f"{system_metrics['mean_recall_at_10']:.4f}")
        
        with col3:
            st.metric("Mean NDCG@10", f"{system_metrics['mean_ndcg_at_10']:.4f}")
        
        # Per-query metrics
        st.subheader("Per-Query Metrics")
        metrics_df = st.session_state.evaluator.get_per_query_metrics()
        st.dataframe(metrics_df, use_container_width=True)
        
        # Metrics visualization
        st.subheader("Metrics Distribution")
        st.plotly_chart(
            st.session_state.evaluator.visualize_metrics_comparison(metrics_df),
            use_container_width=True
        )
        
        # NDCG@K comparison
        st.subheader("NDCG@K Comparison")
        st.plotly_chart(
            st.session_state.evaluator.visualize_ndcg_at_k(k_values=[5, 10, 20]),
            use_container_width=True
        )
    else:
        st.info("No evaluation data available. Please add queries and run evaluation first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Information Retrieval Assignment")
st.sidebar.markdown("Built with Streamlit")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
