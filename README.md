# Hybrid Product Recommendation using Multimodal Similarity

A hybrid recommendation system combining visual and textual information for accurate product similarity search using deep learning and NLP techniques.

## 🚀 Project Overview

This project builds a multimodal product recommendation system by leveraging:

- VGG-16 CNN for extracting visual embeddings from product images.
- TF-IDF for extracting textual embeddings from product descriptions.
- Cosine similarity and Jaccard distance for measuring similarity.
- K-Nearest Neighbors for efficient retrieval.
- KMeans clustering for scalable two-step search.

By combining image and text features, the system delivers highly relevant product recommendations.

## 🔧 Key Features

- Image embeddings using pretrained VGG-16 (ImageNet)
- Text embeddings using TF-IDF vectorizer
- Similarity search using Cosine similarity, Jaccard distance, and KNN
- Scalable two-step retrieval: clustering into 1000 clusters followed by fine-grained search
- Optimal similarity weighting: 0.7 (image) : 0.3 (text)

## 🏗 Technologies Used

- Python  
- TensorFlow / Keras (VGG-16)  
- scikit-learn (TF-IDF, KNN, KMeans)  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

## 🗃 Dataset

The dataset contains:

- Product image files  
- Product descriptions  

Images are resized to 224×224 pixels for VGG-16. Descriptions are preprocessed and vectorized with TF-IDF. Replace the sample dataset with your own catalog for production use.

## 🧑‍💻 Model Pipeline

1. **Image Feature Extraction**  
   - Load and resize the image to 224×224.  
   - Normalize pixel values.  
   - Pass through VGG-16 convolutional base (no top).  
   - Flatten to 512-dimensional feature vectors.  

2. **Text Feature Extraction**  
   - Clean and preprocess descriptions.  
   - Vectorize with `TfidfVectorizer` to obtain sparse embeddings.  

3. **Similarity Computation**  
   - Compute Cosine similarity for image and text vectors.  
   - Compute Jaccard distance for text token overlap.  
   - Use KNN to retrieve top candidates.  
   - Combine modalities:  
     ```
     total_similarity = 0.7 * image_similarity + 0.3 * text_similarity
     ```  

4. **Two-Step Retrieval (Scalability)**  
   - Partition data into 1000 clusters via K-Means.  
   - At query time:  
     1. Identify the nearest cluster to the query.  
     2. Perform a detailed similarity search within that cluster.  


