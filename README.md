# GEN-AI

📈 Stock Market Analysis & News Sentiment Analysis using Generative AI
📌 Project Overview

This project focuses on analyzing stock-market-related news and identifying the sentiment expressed in news articles.

The objective is to build an AI-driven sentiment analysis system that can process financial news, understand its semantic meaning, and classify the news sentiment as Positive, Neutral, or Negative.

The project compares different text embedding techniques and machine learning/deep learning models to identify an effective approach for financial news sentiment classification.

The analysis can support financial analysts and investors by providing structured insights from large volumes of market-related news.

🎯 Problem Statement

Stock prices can be influenced by company performance, market conditions, financial announcements, innovations, collaborations, and news sentiment.

Because a large amount of financial news is generated every day, manually analyzing every article is difficult.

This project uses Artificial Intelligence and Natural Language Processing (NLP) techniques to:

Process stock-market-related news articles
Extract meaningful text representations
Classify news into Positive, Neutral, and Negative sentiment
Compare different embedding techniques
Compare machine learning and neural network models
Identify the better-performing model for sentiment classification
💼 Business Context

Investment firms and financial analysts need to continuously monitor financial news and market sentiment.

An automated sentiment analysis system can help analysts understand the overall tone of news articles and use these insights along with stock-market information for better investment analysis.

The project therefore combines financial news data, stock-price information, NLP, machine learning, and deep learning.

📊 Dataset

The dataset used in this project is:

stock_news.csv

The dataset contains 349 records and 8 columns.

Dataset Features
Column	Description
Date	Date on which the news was released
News	Content of the financial/news article
Open	Opening stock price of the day
High	Highest stock price during the day
Low	Lowest stock price during the day
Close	Adjusted closing stock price
Volume	Number of shares traded during the day
Label	Sentiment polarity of the news
Sentiment Labels
Label	Sentiment
1	Positive
0	Neutral
-1	Negative

The notebook confirms that there are no missing values and no duplicate records in the initial dataset checks.

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Gensim
Word2Vec
Sentence Transformers
BAAI/bge-base-en-v1.5
TensorFlow
Keras
PyTorch
Google Colab

The notebook uses libraries for data preprocessing, visualization, text embeddings, machine learning, deep learning, and model evaluation.

🔄 Project Workflow
Financial News Dataset
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Text Embedding
   ┌────┴────┐
   ↓         ↓
Word2Vec   Sentence Transformer
   ↓         ↓
   └────┬────┘
        ↓
Sentiment Classification
        ↓
 ┌──────┴────────┐
 ↓               ↓
Random Forest   Neural Network
 ↓               ↓
 └──────┬────────┘
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final Recommendation
🔎 Exploratory Data Analysis

The project performs several exploratory analyses to understand the relationship between news sentiment and stock-market variables.

The analysis includes:

Sentiment polarity distribution
Distribution of numerical variables
News-content length analysis
Month-wise analysis
Correlation analysis
Sentiment polarity vs. stock price
Time-series analysis of price
News-length distribution across sentiment categories
🧹 Data Preprocessing

The news dataset is prepared before generating text embeddings.

The preprocessing stage includes preparing the News column and encoding the sentiment labels so that the data can be supplied to machine learning and neural network models.

The target variable is the sentiment Label.

🧠 Text Embeddings

Two different text representation approaches are explored.

1. Word2Vec

Word2Vec is used to convert words in the news articles into numerical vectors.

The project uses a 100-dimensional Word2Vec representation and averages the word vectors to create a fixed-size representation for each news article.

Word2Vec Configuration
vector_size = 100
window = 5
min_count = 1
sg = 0
epochs = 10

The resulting document vectors are used as input features for the classification models.

2. Sentence Transformer

The project also uses the Sentence Transformer model:

BAAI/bge-base-en-v1.5

This approach generates contextual sentence-level embeddings that capture the semantic meaning of the complete news article.

Compared with averaging Word2Vec vectors, Sentence Transformer embeddings provide a richer contextual representation of the news text.

🤖 Machine Learning Models

The project compares two types of classification models.

1. Random Forest

Random Forest is used as a traditional machine-learning classification model.

Two Random Forest approaches are evaluated:

Word2Vec + Random Forest
Sentence Transformer + Random Forest

The Random Forest model is implemented using:

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
2. Neural Network

A neural-network-based classifier is also developed using the generated text embeddings.

Two approaches are evaluated:

Word2Vec + Neural Network
Sentence Transformer + Neural Network

The neural-network models use TensorFlow/Keras with dense layers and dropout.

📏 Model Evaluation

The models are evaluated using:

Accuracy
Recall
Precision
F1 Score
Confusion Matrix

These metrics are used to compare the classification performance of the different embedding-model combinations.

📊 Model Comparison

The notebook compares four model combinations.

Model	Accuracy	Precision	Recall	F1 Score
Word2Vec + Random Forest	21.86%	4.78%	21.86%	7.85%
Word2Vec + Neural Network	49.82%	24.82%	49.82%	33.13%
Sentence Transformer + Random Forest	55.20%	62.62%	55.20%	45.997%
Sentence Transformer + Neural Network	49.82%	24.82%	49.82%	33.13%

These values are from the notebook's model-performance comparison.

🏆 Best Performing Model

Based on the reported comparison, Sentence Transformer + Random Forest achieved the highest accuracy and F1 score among the compared models:

Accuracy: 55.20%
Precision: 62.62%
Recall: 55.20%
F1 Score: 46.00%




💡 Key Findings
Word2Vec

Word2Vec successfully converts words into numerical representations, but averaging word vectors can lose contextual information from the complete sentence.

Sentence Transformer

Sentence Transformer embeddings capture contextual meaning more effectively and provide richer semantic representations for financial news.

Random Forest

Random Forest provides a useful baseline and, in this project, performed particularly well when combined with Sentence Transformer embeddings.

Neural Network

The neural-network approach can learn complex patterns from the generated embeddings, with dropout used to help improve generalization.

These observations are also reflected in the notebook's conclusions.

🚀 Recommendations

Based on the project results:

Use Sentence Transformer embeddings for financial-news sentiment analysis.
Combine contextual embeddings with suitable machine-learning/deep-learning classifiers.
Explore advanced transformer models such as:
BERT
RoBERTa
DeBERTa

These are suggested as possible future improvements in the notebook.

📁 Project Structure
Stock-Market-Analysis/
│
├── Project_I_GenAI.ipynb
├── stock_news.csv
└── README.md
▶️ How to Run
1. Clone the repository
git clone <your-github-repository-url>
2. Open the notebook

Open:

Project_I_GenAI.ipynb

using Google Colab or Jupyter Notebook.

3. Install required libraries
pip install numpy==1.26.4
pip install scikit-learn==1.6.1
pip install scipy==1.13.1
pip install gensim==4.3.3
pip install sentence-transformers==3.4.1
pip install pandas==2.2.2
4. Add the dataset

Place:

stock_news.csv

in the appropriate dataset location.

The original notebook loads the dataset from Google Drive.

5. Run the notebook

Execute the cells sequentially to perform:

Data Loading
      ↓
EDA
      ↓
Preprocessing
      ↓
Word2Vec
      ↓
Sentence Transformer
      ↓
Model Training
      ↓
Evaluation
      ↓
Model Comparison
📌 Project Highlights
Financial news sentiment analysis
NLP-based text representation
Word2Vec embeddings
Contextual Sentence Transformer embeddings
Random Forest classification
Neural Network classification
Confusion matrix analysis
Accuracy, Precision, Recall and F1 evaluation
Comparative model analysis
🎯 Business Impact

An automated financial-news sentiment analysis system can help investment analysts:

Quickly understand large volumes of financial news
Identify positive and negative market sentiment
Reduce manual news-analysis effort
Generate structured sentiment information
Support data-driven investment analysis
Combine news sentiment with stock-market information
⚠️ Disclaimer

This project is intended for educational and analytical purposes.

The sentiment predictions should not be treated as direct financial or investment advice. Stock prices are influenced by many factors beyond news sentiment.
