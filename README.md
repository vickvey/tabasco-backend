# TABASCO FastAPI

## Description

TABASCO FastAPI is a backend API built with FastAPI, designed for detecting intra-domain ambiguities in text files (PDF or TXT). It provides a robust set of endpoints to analyze text, extract linguistic features, and generate summaries based on semantic clustering.

## Features

- **File Upload**: Upload PDF or TXT files for analysis.
- **Noun Extraction**: Extract the top N frequent nouns from uploaded files.
- **Target Word Analysis**: Compute similarity matrices for sentences containing a specified target word.
- **Sentence Clustering**: Cluster sentences based on cosine similarity using BERT embeddings.
- **Context Word Identification**: Identify top-k context words related to a target word in sentences.
- **Summary Generation**: Generate summary text files for each cluster of sentences.
- **Download Summaries**: Download generated summary files for specific clusters.

---
