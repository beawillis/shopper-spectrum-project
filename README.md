Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
Python Version Streamlit App

Table of Contents
Introduction
Problem Statement
Key Features
Data Source
Methodology
Data Preprocessing
RFM Analysis & K-Means Clustering
Item-Based Collaborative Filtering
Deployment (Streamlit)
Installation & Usage
Results & Insights
Technologies Used
Conclusion
Introduction
This project, “Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce,” aims to analyze customer purchasing patterns and build intelligent business solutions using machine learning. It involves data cleaning, preprocessing, and exploratory data analysis (EDA) to identify trends, top-selling products, and customer spending behavior. Customer segmentation is performed using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering, while an Item-Based Collaborative Filtering recommendation system provides personalized product suggestions.

Problem Statement
This project aims to analyze e-commerce transaction data to understand purchasing behavior, identify meaningful customer groups using RFM-based clustering, and generate personalized product recommendations using collaborative filtering. The objective is to improve customer retention, marketing efficiency, inventory planning, and customer experience.

Key Features
Customer Segmentation: Groups customers into distinct behavioral segments (e.g., High-Value, Regular, Occasional, At-Risk) using RFM analysis and K-Means clustering.
Product Recommendation Engine: Provides personalized product suggestions based on item-based collaborative filtering and cosine similarity.
Interactive Streamlit Web Application: A user-friendly interface for real-time customer segmentation classification and product recommendations.
Data Source
The project utilizes the online_retail.csv dataset, which contains transactional data for a UK-based online retail store.

Methodology
Data Preprocessing
Handling of missing CustomerID and Description values.
Removal of cancelled orders (InvoiceNo starting with 'C').
Filtering out records with non-positive Quantity or UnitPrice.
Dropping duplicate rows.
Converting InvoiceDate to datetime objects and CustomerID to integer type.
Creation of TotalAmount (Quantity * UnitPrice) for each transaction.
RFM Analysis & K-Means Clustering
Recency: Days since last purchase.
Frequency: Total number of purchases.
Monetary: Total spend.
RFM features were scaled using StandardScaler to ensure fair weighting in distance-based clustering algorithms.
K-Means Clustering: Applied to the scaled RFM features. The optimal number of clusters (K=4) was determined using the Elbow Method and Silhouette Score analysis.
Segment Mapping: Clusters were semantically mapped to business-relevant segments: "At-Risk", "Regular", "High-Value", and "Occasional".
Item-Based Collaborative Filtering
A customer-product interaction matrix was created, pivoting CustomerID by Description and aggregating Quantity.
Cosine Similarity: Used to compute the similarity between different products based on co-purchase patterns.
The recommendation engine function leverages this similarity matrix to suggest the top N most similar products for a given item.
Deployment (Streamlit)
The solution is deployed as an interactive Streamlit web application, offering two main modules:

Product Recommendation Engine: Users can select a product and receive a list of top 5 recommended items based on collaborative filtering.
Customer Lifecycle Segmentation: Users can input customer operational features (Recency, Frequency, Monetary) to instantly classify their lifecycle value tier.
Installation & Usage
To run this project locally, follow these steps:

Prerequisites
Python 3.8+
npm (Node Package Manager) if using localtunnel for public access in cloud environments like Google Colab.
1. Clone the Repository
git clone https://github.com/your-username/your-repository.git
cd your-repository
2. Install Dependencies
pip install -r requirements.txt
3. Run the Streamlit Application
streamlit run app.py
If running in an environment like Google Colab and you want to share the app publicly, you might use localtunnel:

!streamlit run app.py & npx localtunnel --port 8501
Results & Insights
Customer Segmentation Quality: The K-Means model achieved a Silhouette Coefficient of 0.6162 for 4 clusters, indicating well-separated and distinct customer groups.
Recommendation System Sparsity: Despite a matrix sparsity of 98.40%, the Item-Based Collaborative Filtering effectively identified meaningful product relationships.
Key insights from EDA revealed dominant international markets (Germany, France), peak transaction amounts ( 10− 50), and significant month-over-month revenue growth.
Technologies Used
Python
Pandas
NumPy
Scikit-learn (KMeans, StandardScaler, Cosine Similarity)
Matplotlib
Seaborn
Streamlit
Joblib
Conclusion
This project successfully implemented an integrated analytics system using unsupervised machine learning and collaborative filtering on e-commerce transaction data. By mapping 2,815 unique customers into four non-overlapping behavioral clusters using K-Means, the system achieved a Silhouette Coefficient of 0.5056. This provides a reliable breakdown separating Regular shoppers and At-Risk accounts from hyper-frequent Champions and elite High-Value "Whales," enabling highly targeted lifecycle marketing.

Concurrently, despite facing a 98.69% matrix sparsity challenge, the Item-Based Collaborative Filtering module utilizes Cosine Similarity to uncover deep catalog relationships, serving up accurate, real-time product recommendations. Deployed seamlessly through an interactive Streamlit dashboard, this dual-engine setup transitions the business from speculative retail planning to automated, empirical personalization. The final data product directly helps maximize customer lifetime value (LTV), improve retention strategies, minimize stock irregularities, and optimize marketing return on investment (ROI).
