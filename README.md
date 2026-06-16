# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

End-to-end unsupervised machine learning project that segments e-commerce customers using RFM analysis and recommends similar products using item-based collaborative filtering, deployed as an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📣 Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behaviors. This project analyzes online retail transaction data (2022–2023) to:

1. Uncover patterns in customer purchase behavior through comprehensive EDA
2. Segment customers using **Recency, Frequency, Monetary (RFM)** analysis and unsupervised clustering
3. Build a **product recommendation system** using item-based collaborative filtering
4. Deploy both solutions in a real-time, interactive **Streamlit** application

---

## 📌 Real-World Business Use Cases

- Customer segmentation for targeted marketing campaigns
- Personalized product recommendations on e-commerce platforms
- Identifying at-risk customers for retention programs
- Dynamic pricing strategies based on purchase behavior
- Inventory management and stock optimization based on demand patterns

---

## 🗂️ Dataset

The dataset contains transaction-level records from a UK-based online retail business spanning 2022–2023.

| Column | Description |
|---|---|
| `InvoiceNo` | Unique transaction number (prefix 'C' = cancellation) |
| `StockCode` | Unique product/item code |
| `Description` | Name of the product |
| `Quantity` | Number of units purchased |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit (£) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country of residence |

📎 [Dataset Source](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

---

## 🧠 Problem Type

- **Unsupervised Machine Learning** — Clustering (Customer Segmentation)
- **Collaborative Filtering** — Recommendation System

---

## 🔧 Methodology

### 1. Data Preprocessing
- Removed duplicate rows and parsed `InvoiceDate` to datetime
- Dropped rows with missing `CustomerID` (guest checkouts)
- Excluded cancelled invoices (`InvoiceNo` starting with `C`)
- Removed non-positive `Quantity` and `UnitPrice` values
- Engineered `TotalAmount = Quantity × UnitPrice`

### 2. Exploratory Data Analysis
15 charts following the **Univariate → Bivariate → Multivariate (UBM)** framework, covering:
- Quantity, price, and monetary distributions
- Transaction volume and revenue by country
- Top-selling and top-revenue products
- Monthly revenue trend and seasonality
- Day-of-week purchasing patterns
- Customer spend distribution and Pareto (80/20) analysis
- RFM feature distributions and correlation analysis

### 3. Hypothesis Testing
Statistical validation using Mann-Whitney U and Kruskal-Wallis H tests on:
- UK vs. non-UK customer spend differences
- Recency differences between high- and low-frequency customers
- Seasonality of monthly transaction volume

### 4. RFM Feature Engineering & Customer Segmentation
- Computed **Recency**, **Frequency**, and **Monetary** values per customer
- Applied log-transformation and IQR-based outlier capping
- Standardized features using `StandardScaler`
- Compared three clustering algorithms:

| Model | Method | Notes |
|---|---|---|
| **KMeans** ✅ | Elbow Method + Silhouette Score → k=4 | Best Silhouette Score, full customer coverage — **selected as final model** |
| Agglomerative Clustering | Ward linkage, dendrogram-guided | Comparable performance, more computationally expensive |
| DBSCAN | k-distance plot for `eps` selection | Flags noise/outliers, but leaves some customers unsegmented |

- Final segments: **High-Value**, **Regular**, **Occasional**, **At-Risk**

### 5. Product Recommendation System
- Built a CustomerID × Product pivot matrix from purchase history
- Computed **cosine similarity** between products (item-based collaborative filtering)
- Returns the top 5 most similar products for any queried product name

---

## 📱 Streamlit App Features

### 🎯 Module 1 — Product Recommendation
- Text input for a product name
- "Get Recommendations" button
- Displays the top 5 similar products with similarity scores

### 🎯 Module 2 — Customer Segmentation
- Numeric inputs for **Recency**, **Frequency**, and **Monetary**
- "Predict Cluster" button
- Displays the predicted customer segment (High-Value / Regular / Occasional / At-Risk)

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`

**Techniques:** Data Cleaning, Feature Engineering, EDA, RFM Analysis, Hypothesis Testing, KMeans/Agglomerative/DBSCAN Clustering, Collaborative Filtering, Cosine Similarity, PCA, StandardScaler, Model Persistence

---

## 📁 Project Structure

```
shopper-spectrum-customer-segmentation/
│
├── Shopper_Spectrum_Customer_Segmentation.ipynb   # Full analysis & modeling notebook
├── app.py                                         # Streamlit application
├── kmeans_rfm_model.pkl                           # Trained KMeans model
├── rfm_scaler.pkl                                 # Fitted StandardScaler
├── item_similarity_matrix.pkl                     # Cosine similarity matrix
├── segment_labels.pkl                             # Cluster → segment label mapping
├── cluster_features.pkl                           # Feature order reference
├── requirements.txt                               # Project dependencies
└── README.md                                      # Project documentation
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/harshvardhan/shopper-spectrum-customer-segmentation.git
cd shopper-spectrum-customer-segmentation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Jupyter Notebook** (to reproduce the analysis and regenerate model artifacts)
```bash
jupyter notebook Shopper_Spectrum_Customer_Segmentation.ipynb
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

---

## 📊 Key Results

- **~500K+** transaction records cleaned and processed
- **15** EDA visualizations revealing UK market dominance, Q4 seasonality, and an 80/20 customer revenue concentration
- **4 customer segments** identified via KMeans (k=4) with the best Silhouette Score among all models tested
- **Item-based collaborative filtering** delivers real-time top-5 product recommendations
- Fully serialized model pipeline ready for production deployment

---

## 🚀 Future Work

- Incorporate a hybrid recommendation approach (content-based + collaborative filtering)
- Add RFM-based customer lifetime value (CLV) prediction
- Automate periodic model retraining on new transaction data
- Extend segmentation with additional behavioral features (e.g., return rate, category preference)

---

## 👤 Author

**Harshvardhan**
Computer Science (AI & ML) — IPS Academy, Indore

---

## 📄 License

This project is licensed under the MIT License.
