# Twitter Emotion Topic Modelling with BERTopic

**ICT202 Machine Learning | Assignment 2**
**Murdoch University Dubai**
**Parth Kotak | 35562858**

## What This Project Does

Ever wondered what people actually talk about when they're happy, sad, or angry on Twitter? This project digs into that question using BERTopic, a modern topic modelling framework built on top of transformers. Instead of relying on older bag-of-words approaches, the pipeline uses sentence-level embeddings to capture what tweets really mean in context, then clusters them into coherent topics.

The goal is to let the data speak for itself. No predefined categories, no manual labelling. The model discovers topics on its own, and then we compare what it found against the six known emotion labels to see how they line up.

## The Dataset

The data comes from the [Twitter Emotion Classification Dataset](https://www.kaggle.com/datasets/aadyasingh55/twitter-emotion-classification-dataset) on Kaggle. It contains 416,809 English tweets, each tagged with one of six emotions: sadness, joy, love, anger, fear, and surprise. Joy makes up about a third of the data, while surprise is the smallest slice at roughly 3.6%.

Running BERTopic on all 400K+ tweets would take a while, so the notebook pulls a stratified sample of 30,000 tweets. That keeps the original emotion proportions intact and is still 15 times the minimum requirement of 2,000. The dataset downloads automatically when you run the notebook, so there's nothing to set up manually.

## How the Pipeline Works
```
Raw Tweets
    |
    v
Light Preprocessing (URL, mention, hashtag removal)
    |
    v
Sentence Embeddings (all-MiniLM-L6-v2)
    |
    v
Dimensionality Reduction (UMAP)
    |
    v
Density-Based Clustering (HDBSCAN)
    |
    v
Topic Representation (c-TF-IDF)
    |
    v
Evaluation & Visualization
```

A quick note on preprocessing: because the model uses transformer embeddings that understand full sentences, heavy-handed steps like stemming and stop-word removal would actually strip away useful context. So we only clean out the Twitter noise (URLs, @mentions, hashtag symbols, RT prefixes) and leave the natural language intact.

## Key Components

| Component | Method | Details |
|-----------|--------|---------|
| Embedding | Sentence-BERT | `all-MiniLM-L6-v2`, 384-dimensional vectors |
| Dimensionality Reduction | UMAP | n_neighbors=15, n_components=5, cosine metric |
| Clustering | HDBSCAN | min_cluster_size=30, min_samples=10 |
| Topic Representation | c-TF-IDF | Class-based TF-IDF for keyword extraction |

## How the Model is Evaluated

Three metrics are used to judge whether the topics are any good:

- **Topic Coherence (C_v)** checks if the top words within each topic actually belong together semantically. Scores above 0.4 are generally solid.
- **Topic Diversity** looks at how much overlap there is between topics. A high score means each topic has its own distinct vocabulary rather than repeating the same words.
- **Silhouette Score** measures how cleanly the documents separate into their assigned clusters. Anything above 0 means the clustering is doing better than random.

## Visualizations

The notebook produces six visualizations that each tell a different part of the story:

1. **Intertopic Distance Map** showing how topics relate to each other in 2D space
2. **Top Words Bar Chart** for the most representative words per topic
3. **Topic Hierarchy Dendrogram** revealing which topics are close cousins
4. **Topic Similarity Heatmap** with pairwise similarity scores
5. **2D Document Scatter Plot** where every dot is a tweet, coloured by topic
6. **Topic vs Emotion Heatmap** showing how the unsupervised topics map onto the six emotion labels (this one is probably the most interesting)

## How to Run It

### Google Colab (Recommended)

1. Upload `ICT202_Assignment2Submission_BERTopic.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set the runtime to **T4 GPU** (Runtime > Change runtime type) for faster embeddings
3. Run all cells top to bottom
4. The dataset downloads itself from Kaggle automatically

### Running Locally
```bash
git clone https://github.com/parthbtw/ict202-bertopic-tweet-analysis.git
cd ict202-bertopic-tweet-analysis
pip install -r requirements.txt
jupyter notebook ICT202_Assignment2Submission_BERTopic.ipynb
```

## Project Structure
```
ict202-bertopic-tweet-analysis/
├── ICT202_Assignment2Submission_BERTopic.ipynb  # Main analysis notebook
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
└── figures/                                     # Generated after running the notebook
```

## Built With

Python, BERTopic, Sentence-Transformers, UMAP, HDBSCAN, Gensim, Scikit-learn, Plotly, Seaborn, Matplotlib, Pandas, NumPy

## References

1. M. Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure," arXiv:2203.05794, 2022.
2. L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," arXiv:1802.03426, 2018.
3. L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," JOSS, vol. 2, no. 11, 2017.
4. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.

## License

This project was developed for the ICT202 Machine Learning unit at Murdoch University Dubai.
