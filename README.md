# Information Retrieval System - Assignment 2

A comprehensive Streamlit-based Information Retrieval system implementing the complete IR lifecycle including web crawling, text preprocessing, indexing, search, ranking, recommendation, and evaluation.

## Features

### 1. Web Crawling
- Configurable crawling depth and multiple seed sources
- Duplicate URL and document detection
- Metadata extraction separate from document content
- URL graph construction for link analysis

### 2. Text Preprocessing and Mining
- Scalable text preprocessing pipeline
- Keyword extraction using TF-IDF
- Document profiling with comprehensive statistics
- Topic modeling using LDA
- Document clustering with K-means
- Comparative analysis of preprocessing strategies
- Interactive visualizations

### 3. Web Searching
- Intelligent query processing
- Ranked document retrieval
- Multiple ranking algorithms:
  - TF-IDF
  - PageRank
  - HITS (Hubs and Authorities)
- Advanced search with filters
- Query optimization

### 4. Recommender System
- Content-based recommendation
- Collaborative filtering (user-based and item-based)
- Hybrid recommendation approach
- Top-K recommendations with similarity scores

### 5. Evaluation Metrics
- Precision, Recall, F1-score
- Precision@K, Recall@K
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Comparative analysis with visualizations

### 6. Streamlit Interface
- Dashboard with system overview
- Web crawling interface
- Text preprocessing panel
- Index management
- Search interface
- Ranking visualization
- Recommendation panel
- Evaluation dashboard
- Performance analytics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Navigate to the project directory:**
   ```bash
   cd c:/Users/Dell/Desktop/acer_Desktop_BKP/VVIMP_Balaji/MTECH-BITS/002_Sem/IR/Assignment2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will automatically open in your default web browser at `http://localhost:8501`

## Dependencies

- streamlit==1.28.0
- beautifulsoup4==4.12.2
- requests==2.31.0
- lxml==4.9.3
- nltk==3.8.1
- scikit-learn==1.3.0
- pandas==2.0.3
- numpy==1.24.3
- plotly==5.17.0
- networkx==3.1
- whoosh==2.7.4
- tqdm==4.66.1
- python-dateutil==2.8.2

## Usage Guide

### Quick Start

1. **Load Sample Data:**
   - Go to the Dashboard page
   - Click "Load Sample Data" to load 5 sample documents
   - Click "Process Documents" to preprocess the text
   - Click "Create Index" to build the search index

2. **Web Crawling:**
   - Navigate to "Web Crawling" page
   - Enter seed URLs (one per line)
   - Configure max depth, max pages, and domain restriction
   - Click "Start Crawling"
   - View crawled documents and statistics

3. **Text Preprocessing:**
   - Navigate to "Text Preprocessing" page
   - Configure preprocessing options
   - View corpus statistics and visualizations
   - Run topic modeling and clustering

4. **Search:**
   - Navigate to "Web Search" page
   - Enter search query
   - Select ranking method (TF-IDF, PageRank, HITS)
   - View ranked results with scores

5. **Ranking Visualization:**
   - Navigate to "Ranking Visualization" page
   - View PageRank scores
   - View HITS authority scores
   - Visualize URL graph structure

6. **Recommendations:**
   - Navigate to "Recommendation Panel" page
   - Select recommendation approach (content-based or hybrid)
   - Choose recommendation method (by document or by query)
   - View Top-K recommendations with similarity scores

7. **Evaluation:**
   - Navigate to "Evaluation Dashboard" page
   - Add queries with relevant documents
   - Run search and evaluate
   - View precision, recall, F1, MAP, MRR, NDCG
   - Analyze precision-recall curves

8. **Performance Analytics:**
   - Navigate to "Performance Analytics" page
   - View system-wide metrics
   - Compare per-query performance
   - Analyze metrics distribution

## Project Structure

```
Assignment2/
├── app.py                      # Main Streamlit application
├── web_crawler.py              # Web crawling module
├── text_preprocessing.py       # Text preprocessing and mining
├── search_engine.py            # Search engine with PageRank/HITS
├── recommender_system.py       # Recommender systems
├── evaluation_metrics.py       # IR evaluation metrics
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── documents.json              # Crawled documents (generated)
├── metadata.csv                # Document metadata (generated)
├── url_graph.json              # URL link graph (generated)
└── index_dir/                  # Whoosh search index (generated)
```

## Module Descriptions

### web_crawler.py
Implements a web crawler with:
- Configurable depth and multiple seed sources
- Duplicate URL detection using normalization
- Duplicate document detection using content hashing
- Metadata extraction (title, description, keywords)
- URL graph construction for link analysis

### text_preprocessing.py
Provides text mining framework with:
- Text cleaning and normalization
- Tokenization and stopword removal
- Lemmatization
- TF-IDF feature extraction
- Keyword extraction
- Document profiling
- Topic modeling (LDA)
- Document clustering (K-means)
- Comparative preprocessing analysis

### search_engine.py
Implements search functionality with:
- Whoosh-based full-text indexing
- TF-IDF ranking
- PageRank algorithm implementation
- HITS algorithm implementation
- Advanced search with filters
- Ranking visualization
- Graph visualization

### recommender_system.py
Provides recommendation approaches:
- Content-based filtering using TF-IDF similarity
- Collaborative filtering (user-based and item-based)
- Hybrid recommendation combining multiple approaches
- Top-K recommendations with scores
- Recommendation visualization

### evaluation_metrics.py
Implements IR evaluation metrics:
- Precision, Recall, F1-score
- Precision@K, Recall@K
- Average Precision (AP)
- Mean Average Precision (MAP)
- Reciprocal Rank (RR)
- Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG)
- Normalized DCG (NDCG)
- Precision-Recall curves
- Metrics comparison visualization

## Inference and Discussion

### 1. Poor Ranking of Relevant Documents
**Possible causes:**
- Inadequate term weighting in TF-IDF
- Lack of query expansion
- Insufficient document length normalization
- Poor handling of synonyms
- Ineffective ranking algorithm parameters

**Improvements:**
- Implement query expansion with synonyms
- Use BM25 ranking instead of TF-IDF
- Add document length normalization
- Incorporate user feedback for relevance
- Tune ranking algorithm parameters

### 2. Impact of Duplicate Documents
**Effects:**
- **Indexing:** Increased index size, redundant terms
- **Ranking:** Skewed relevance scores, bias toward duplicates
- **Recommendation:** Reduced diversity, repetitive suggestions
- **Evaluation:** Inflated precision metrics

**Mitigation:**
- Content hashing for duplicate detection
- Near-duplicate detection using shingling
- URL normalization
- Document clustering to identify similar content
- Deduplication before indexing

### 3. Content-based vs Collaborative Recommendation

**Content-based:**
- Preferable when: User preferences are known, item features are available, cold-start for new users
- Advantages: No cold-start for items, transparent recommendations
- Disadvantages: Limited novelty, requires feature extraction

**Collaborative:**
- Preferable when: User interaction data is available, community wisdom is valuable
- Advantages: Can discover unexpected items, no feature engineering needed
- Disadvantages: Cold-start problem, sparsity issues

**Hybrid approach** combines strengths of both for better performance.

### 4. Integration Benefits
The integration of all components creates a comprehensive IR system:
- **Crawling** provides fresh, relevant content
- **Text mining** extracts structured features
- **Indexing** enables efficient retrieval
- **Search** provides ranked results
- **Ranking** improves result quality
- **Recommendation** enhances user experience
- **Evaluation** ensures system effectiveness

Each component builds on the previous, creating a pipeline that transforms raw web content into actionable insights and recommendations.

### 5. Key Learnings
- Importance of preprocessing for IR effectiveness
- Trade-offs between different ranking algorithms
- Value of multiple evaluation metrics
- Impact of data quality on system performance
- Benefits of hybrid approaches in recommendation
- Need for continuous evaluation and improvement

## Troubleshooting

### Common Issues

1. **NLTK data not found:**
   - The system automatically downloads required NLTK data on first run
   - Ensure internet connection is available

2. **Crawling errors:**
   - Some websites may block crawlers
   - Try different seed URLs
   - Increase timeout values in web_crawler.py

3. **Index creation fails:**
   - Ensure documents are processed first
   - Check that index_dir doesn't have permission issues
   - Delete existing index_dir and recreate

4. **Memory issues:**
   - Reduce max_pages when crawling
   - Reduce max_features in TF-IDF vectorization
   - Process documents in batches

## Submission Components

This project includes all required submission components:

1. ✅ Streamlit application code (app.py and supporting modules)
2. ✅ Dataset (sample data built-in, supports external crawling)
3. ✅ Report (comprehensive README with implementation details)
4. ✅ Demo evidence (screenshots can be captured from running app)
5. ✅ README file (this document with setup instructions)

## License

This project is created for educational purposes as part of the Information Retrieval course assignment.

## Contact

For any queries or clarifications, please use the Taxila Discussion Forum.
