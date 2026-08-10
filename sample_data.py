"""Bundled dataset for the Information Retrieval system.

Provides a reproducible 24-document corpus with metadata, hyperlink graph,
user ratings, graded relevance judgements for 7 queries, and near-duplicate
samples for testing duplicate detection.
"""

import hashlib


def get_dataset() -> dict:
    """Return the complete bundled dataset.

    Returns:
        dict with keys: documents, metadata, near_duplicates, url_graph,
                       labels, ratings, queries
    """
    documents = _get_documents()
    metadata = _get_metadata()
    near_duplicates = _get_near_duplicates()
    url_graph = _get_url_graph()
    labels = _get_labels()
    ratings = _get_ratings()
    queries = _get_queries()

    return {
        "documents": documents,
        "metadata": metadata,
        "near_duplicates": near_duplicates,
        "url_graph": url_graph,
        "labels": labels,
        "ratings": ratings,
        "queries": queries,
    }


def _get_documents() -> list:
    """Return the 24 document corpus."""
    docs = [
        {
            "url": "https://example.com/ml/intro",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning techniques. ML models can make predictions or decisions without being explicitly programmed.",
        },
        {
            "url": "https://example.com/ml/supervised",
            "content": "Supervised learning uses labeled data to train models. Common algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks. The model learns to map inputs to outputs based on training examples.",
        },
        {
            "url": "https://example.com/ml/unsupervised",
            "content": "Unsupervised learning finds patterns in unlabeled data. Key techniques include clustering, dimensionality reduction, and association rule learning. Popular algorithms are K-means, hierarchical clustering, PCA, and autoencoders.",
        },
        {
            "url": "https://example.com/dl/intro",
            "content": "Deep learning uses neural networks with multiple layers to model complex patterns in data. It has achieved remarkable success in image recognition, natural language processing, and speech recognition. Deep networks can automatically learn feature representations.",
        },
        {
            "url": "https://example.com/dl/cnn",
            "content": "Convolutional Neural Networks are specialized for processing grid-like data such as images. They use convolutional layers to detect local patterns, pooling layers to reduce spatial dimensions, and fully connected layers for classification. CNNs are widely used in computer vision.",
        },
        {
            "url": "https://example.com/dl/rnn",
            "content": "Recurrent Neural Networks are designed for sequential data like text and time series. They maintain internal state to capture temporal dependencies. Variants include LSTM and GRU which address the vanishing gradient problem. RNNs power many NLP applications.",
        },
        {
            "url": "https://example.com/nlp/intro",
            "content": "Natural language processing enables computers to understand and generate human language. Applications include machine translation, sentiment analysis, chatbots, and text summarization. Modern NLP heavily relies on transformer architectures.",
        },
        {
            "url": "https://example.com/nlp/transformers",
            "content": "Transformer models revolutionized NLP with self-attention mechanisms. They can process entire sequences in parallel, enabling better handling of long-range dependencies. BERT, GPT, and T5 are prominent transformer architectures used for various language tasks.",
        },
        {
            "url": "https://example.com/nlp/embeddings",
            "content": "Word embeddings represent words as dense vectors in continuous space. Word2Vec, GloVe, and FastText are popular embedding methods. Contextual embeddings from transformers capture word meaning based on surrounding context, enabling better semantic understanding.",
        },
        {
            "url": "https://example.com/cv/intro",
            "content": "Computer vision allows machines to interpret and understand visual information from the world. Key tasks include object detection, image classification, facial recognition, and image segmentation. Deep learning has dramatically advanced computer vision capabilities.",
        },
        {
            "url": "https://example.com/cv/detection",
            "content": "Object detection locates and classifies objects within images. YOLO, Faster R-CNN, and SSD are popular detection architectures. These models use bounding boxes to identify object positions and class labels to categorize what they detect.",
        },
        {
            "url": "https://example.com/cv/segmentation",
            "content": "Image segmentation partitions images into meaningful regions. Semantic segmentation assigns class labels to each pixel, while instance segmentation distinguishes between individual objects. U-Net and Mask R-CNN are widely used segmentation models.",
        },
        {
            "url": "https://example.com/rl/intro",
            "content": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. The agent learns through trial and error interaction with an environment. RL has been successfully applied to game playing, robotics, and autonomous systems.",
        },
        {
            "url": "https://example.com/rl/qlearning",
            "content": "Q-learning is a model-free reinforcement learning algorithm. It learns the value of state-action pairs through exploration and exploitation. The Q-table stores expected future rewards for each action in each state, guiding the agent toward optimal policies.",
        },
        {
            "url": "https://example.com/rl/policy",
            "content": "Policy gradient methods directly optimize the policy function rather than value functions. REINFORCE and actor-critic methods are common approaches. These algorithms are particularly useful for continuous action spaces and stochastic environments.",
        },
        {
            "url": "https://example.com/data/intro",
            "content": "Data preprocessing is crucial for machine learning success. Steps include data cleaning, feature scaling, encoding categorical variables, and handling missing values. Proper preprocessing ensures models receive high-quality input data.",
        },
        {
            "url": "https://example.com/data/feature",
            "content": "Feature engineering creates new features from raw data to improve model performance. Techniques include polynomial features, interaction terms, binning, and domain-specific transformations. Good features can make simple models perform as well as complex ones.",
        },
        {
            "url": "https://example.com/data/selection",
            "content": "Feature selection reduces dimensionality by keeping only the most relevant features. Methods include filter methods, wrapper methods, and embedded techniques like LASSO. Feature selection improves model interpretability and reduces overfitting.",
        },
        {
            "url": "https://example.com/eval/metrics",
            "content": "Model evaluation uses metrics to assess performance. Classification metrics include accuracy, precision, recall, F1-score, and ROC-AUC. Regression metrics use MSE, MAE, and R-squared. Cross-validation provides reliable performance estimates.",
        },
        {
            "url": "https://example.com/eval/validation",
            "content": "Validation techniques prevent overfitting and ensure generalization. Train-test split, k-fold cross-validation, and leave-one-out validation are common approaches. Proper validation gives confidence in model performance on unseen data.",
        },
        {
            "url": "https://example.com/eval/bias",
            "content": "Bias and variance trade-off is fundamental in machine learning. High bias causes underfitting, while high variance causes overfitting. Regularization, ensemble methods, and proper model complexity help balance this trade-off.",
        },
        {
            "url": "https://example.com/ensemble/intro",
            "content": "Ensemble methods combine multiple models to improve performance. Bagging reduces variance by training models on bootstrap samples. Boosting reduces bias by sequentially training models that focus on difficult examples. Random forests and gradient boosting are popular ensembles.",
        },
        {
            "url": "https://example.com/ensemble/randomforest",
            "content": "Random forests are an ensemble of decision trees trained on bootstrap samples with random feature selection. They provide excellent performance with minimal tuning. Random forests handle both classification and regression tasks and include built-in feature importance.",
        },
        {
            "url": "https://example.com/ensemble/boosting",
            "content": "Gradient boosting machines build models sequentially, each correcting errors of previous ones. XGBoost, LightGBM, and CatBoost are efficient implementations. Boosting often achieves state-of-the-art results on structured data problems.",
        },
    ]
    
    # Add hashes to documents
    for doc in docs:
        doc["hash"] = hashlib.md5(doc["content"].encode()).hexdigest()
    
    return docs


def _get_metadata() -> list:
    """Return metadata for the documents."""
    return [
        {"title": "Introduction to Machine Learning", "url": "https://example.com/ml/intro"},
        {"title": "Supervised Learning Algorithms", "url": "https://example.com/ml/supervised"},
        {"title": "Unsupervised Learning Techniques", "url": "https://example.com/ml/unsupervised"},
        {"title": "Deep Learning Overview", "url": "https://example.com/dl/intro"},
        {"title": "Convolutional Neural Networks", "url": "https://example.com/dl/cnn"},
        {"title": "Recurrent Neural Networks", "url": "https://example.com/dl/rnn"},
        {"title": "Natural Language Processing", "url": "https://example.com/nlp/intro"},
        {"title": "Transformer Models", "url": "https://example.com/nlp/transformers"},
        {"title": "Word Embeddings", "url": "https://example.com/nlp/embeddings"},
        {"title": "Computer Vision Introduction", "url": "https://example.com/cv/intro"},
        {"title": "Object Detection", "url": "https://example.com/cv/detection"},
        {"title": "Image Segmentation", "url": "https://example.com/cv/segmentation"},
        {"title": "Reinforcement Learning", "url": "https://example.com/rl/intro"},
        {"title": "Q-Learning Algorithm", "url": "https://example.com/rl/qlearning"},
        {"title": "Policy Gradient Methods", "url": "https://example.com/rl/policy"},
        {"title": "Data Preprocessing", "url": "https://example.com/data/intro"},
        {"title": "Feature Engineering", "url": "https://example.com/data/feature"},
        {"title": "Feature Selection", "url": "https://example.com/data/selection"},
        {"title": "Model Evaluation Metrics", "url": "https://example.com/eval/metrics"},
        {"title": "Validation Techniques", "url": "https://example.com/eval/validation"},
        {"title": "Bias-Variance Trade-off", "url": "https://example.com/eval/bias"},
        {"title": "Ensemble Methods", "url": "https://example.com/ensemble/intro"},
        {"title": "Random Forests", "url": "https://example.com/ensemble/randomforest"},
        {"title": "Gradient Boosting", "url": "https://example.com/ensemble/boosting"},
    ]


def _get_near_duplicates() -> list:
    """Return near-duplicate documents for testing duplicate detection."""
    return [
        {
            "url": "https://example.com/ml/intro-dup",
            "content": "Machine learning is a branch of artificial intelligence that enables systems to learn from data. It encompasses supervised learning, unsupervised learning, and reinforcement learning approaches. ML algorithms can make predictions without explicit programming.",
        },
        {
            "url": "https://example.com/dl/intro-dup",
            "content": "Deep learning utilizes multi-layer neural networks to capture complex patterns in data. It has shown exceptional performance in image recognition, natural language processing, and speech recognition tasks. Deep networks automatically learn feature representations from raw data.",
        },
        {
            "url": "https://example.com/nlp/intro-dup",
            "content": "Natural language processing allows computers to comprehend and generate human language. Key applications include machine translation, sentiment analysis, conversational agents, and text summarization. Contemporary NLP heavily depends on transformer-based architectures.",
        },
    ]


def _get_url_graph() -> dict:
    """Return the hyperlink graph structure."""
    return {
        "https://example.com/ml/intro": [
            "https://example.com/ml/supervised",
            "https://example.com/ml/unsupervised",
            "https://example.com/dl/intro",
        ],
        "https://example.com/ml/supervised": [
            "https://example.com/dl/intro",
            "https://example.com/ensemble/intro",
        ],
        "https://example.com/ml/unsupervised": [
            "https://example.com/dl/intro",
            "https://example.com/data/feature",
        ],
        "https://example.com/dl/intro": [
            "https://example.com/dl/cnn",
            "https://example.com/dl/rnn",
            "https://example.com/cv/intro",
            "https://example.com/nlp/intro",
        ],
        "https://example.com/dl/cnn": [
            "https://example.com/cv/detection",
            "https://example.com/cv/segmentation",
        ],
        "https://example.com/dl/rnn": [
            "https://example.com/nlp/transformers",
            "https://example.com/nlp/embeddings",
        ],
        "https://example.com/nlp/intro": [
            "https://example.com/nlp/transformers",
            "https://example.com/nlp/embeddings",
        ],
        "https://example.com/nlp/transformers": [
            "https://example.com/dl/intro",
        ],
        "https://example.com/nlp/embeddings": [
            "https://example.com/data/feature",
        ],
        "https://example.com/cv/intro": [
            "https://example.com/cv/detection",
            "https://example.com/cv/segmentation",
        ],
        "https://example.com/cv/detection": [
            "https://example.com/dl/cnn",
        ],
        "https://example.com/cv/segmentation": [
            "https://example.com/dl/cnn",
        ],
        "https://example.com/rl/intro": [
            "https://example.com/rl/qlearning",
            "https://example.com/rl/policy",
        ],
        "https://example.com/rl/qlearning": [
            "https://example.com/ensemble/randomforest",
        ],
        "https://example.com/rl/policy": [
            "https://example.com/ensemble/boosting",
        ],
        "https://example.com/data/intro": [
            "https://example.com/data/feature",
            "https://example.com/data/selection",
        ],
        "https://example.com/data/feature": [
            "https://example.com/ensemble/intro",
        ],
        "https://example.com/data/selection": [
            "https://example.com/eval/metrics",
        ],
        "https://example.com/eval/metrics": [
            "https://example.com/eval/validation",
            "https://example.com/eval/bias",
        ],
        "https://example.com/eval/validation": [
            "https://example.com/ensemble/intro",
        ],
        "https://example.com/eval/bias": [
            "https://example.com/ensemble/randomforest",
        ],
        "https://example.com/ensemble/intro": [
            "https://example.com/ensemble/randomforest",
            "https://example.com/ensemble/boosting",
        ],
        "https://example.com/ensemble/randomforest": [],
        "https://example.com/ensemble/boosting": [],
    }


def _get_labels() -> list:
    """Return document category labels."""
    return [
        "ml", "ml", "ml", "dl", "dl", "dl", "nlp", "nlp", "nlp", "cv",
        "cv", "cv", "rl", "rl", "rl", "data", "data", "data", "eval",
        "eval", "eval", "ensemble", "ensemble", "ensemble",
    ]


def _get_ratings() -> list:
    """Return user-item ratings for collaborative filtering."""
    return [
        {"user_id": "user1", "doc_id": 0, "rating": 5},
        {"user_id": "user1", "doc_id": 3, "rating": 4},
        {"user_id": "user1", "doc_id": 6, "rating": 5},
        {"user_id": "user1", "doc_id": 9, "rating": 3},
        {"user_id": "user1", "doc_id": 12, "rating": 4},
        {"user_id": "user1", "doc_id": 15, "rating": 5},
        {"user_id": "user2", "doc_id": 0, "rating": 4},
        {"user_id": "user2", "doc_id": 1, "rating": 5},
        {"user_id": "user2", "doc_id": 4, "rating": 4},
        {"user_id": "user2", "doc_id": 7, "rating": 5},
        {"user_id": "user2", "doc_id": 10, "rating": 3},
        {"user_id": "user2", "doc_id": 16, "rating": 4},
        {"user_id": "user3", "doc_id": 1, "rating": 4},
        {"user_id": "user3", "doc_id": 2, "rating": 5},
        {"user_id": "user3", "doc_id": 5, "rating": 4},
        {"user_id": "user3", "doc_id": 8, "rating": 5},
        {"user_id": "user3", "doc_id": 17, "rating": 3},
        {"user_id": "user3", "doc_id": 21, "rating": 4},
        {"user_id": "user4", "doc_id": 3, "rating": 5},
        {"user_id": "user4", "doc_id": 4, "rating": 4},
        {"user_id": "user4", "doc_id": 5, "rating": 5},
        {"user_id": "user4", "doc_id": 18, "rating": 4},
        {"user_id": "user4", "doc_id": 22, "rating": 5},
        {"user_id": "user5", "doc_id": 6, "rating": 4},
        {"user_id": "user5", "doc_id": 7, "rating": 5},
        {"user_id": "user5", "doc_id": 8, "rating": 4},
        {"user_id": "user5", "doc_id": 19, "rating": 5},
        {"user_id": "user5", "doc_id": 23, "rating": 3},
        {"user_id": "user6", "doc_id": 9, "rating": 5},
        {"user_id": "user6", "doc_id": 10, "rating": 4},
        {"user_id": "user6", "doc_id": 11, "rating": 5},
        {"user_id": "user6", "doc_id": 20, "rating": 4},
        {"user_id": "user6", "doc_id": 21, "rating": 5},
        {"user_id": "user7", "doc_id": 12, "rating": 4},
        {"user_id": "user7", "doc_id": 13, "rating": 5},
        {"user_id": "user7", "doc_id": 14, "rating": 4},
        {"user_id": "user7", "doc_id": 15, "rating": 5},
        {"user_id": "user7", "doc_id": 22, "rating": 4},
        {"user_id": "user8", "doc_id": 0, "rating": 3},
        {"user_id": "user8", "doc_id": 6, "rating": 4},
        {"user_id": "user8", "doc_id": 12, "rating": 5},
        {"user_id": "user8", "doc_id": 18, "rating": 4},
        {"user_id": "user8", "doc_id": 23, "rating": 5},
    ]


def _get_queries() -> list:
    """Return queries with graded relevance judgements."""
    return [
        {
            "query_id": "q1",
            "query_text": "machine learning algorithms",
            "relevant_doc_ids": {0, 1, 2},
            "graded_relevance": {0: 2, 1: 2, 2: 1},
        },
        {
            "query_id": "q2",
            "query_text": "deep learning neural networks",
            "relevant_doc_ids": {3, 4, 5},
            "graded_relevance": {3: 2, 4: 2, 5: 1},
        },
        {
            "query_id": "q3",
            "query_text": "natural language processing",
            "relevant_doc_ids": {6, 7, 8},
            "graded_relevance": {6: 2, 7: 2, 8: 1},
        },
        {
            "query_id": "q4",
            "query_text": "computer vision",
            "relevant_doc_ids": {9, 10, 11},
            "graded_relevance": {9: 2, 10: 2, 11: 1},
        },
        {
            "query_id": "q5",
            "query_text": "reinforcement learning",
            "relevant_doc_ids": {12, 13, 14},
            "graded_relevance": {12: 2, 13: 1, 14: 1},
        },
        {
            "query_id": "q6",
            "query_text": "data preprocessing feature engineering",
            "relevant_doc_ids": {15, 16, 17},
            "graded_relevance": {15: 2, 16: 2, 17: 1},
        },
        {
            "query_id": "q7",
            "query_text": "model evaluation ensemble methods",
            "relevant_doc_ids": {18, 19, 20, 21, 22, 23},
            "graded_relevance": {18: 2, 19: 1, 20: 1, 21: 2, 22: 2, 23: 2},
        },
    ]
