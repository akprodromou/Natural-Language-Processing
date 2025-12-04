# Natural Language Processing

A repository with simple practice projects on techniques for text processing and tokenization.

## Regular Expressions and NLTK

The exercises in this part use Python's re library (for regular expressions) and the NLTK gutenberg corpus for classic text analysis tasks.

## Byte-Pair Encoding (BPE)

This section implements the Byte-Pair Encoding (BPE) algorithm from scratch. BPE is a data compression and text tokenization technique used in modern Natural Language Processing (NLP) models.

## Embeddings and Sentiment Classification

This project deals with two core NLP methods: N-gram Language Models and Word Embeddings. It uses the IMDB review dataset for sentiment classification.

In the first part, we build two separate Trigram ($N=3$) LMs with Laplace Smoothing. The LMs are used to measure sentence Perplexity, indicating how well each model fits new text.

The second part implements Cosine Similarity to compare embedding vectors. This converts an entire review into a single, dense Embedding using pre-trained fastText vectors.