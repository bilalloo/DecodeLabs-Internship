**Name:** _Muhammad Bilal_

**Department:** _Data Science_

# **Decodelabs Data Science Internship**

## **Description:**

### **Project 1: Advanced EDA & Feature Engineering**
* **Dataset Used:** `mymoviedb.csv` – A movie metadata dataset containing information such as release dates, popularity, vote counts, and genres. 
* **What I Did:** I transformed raw, chaotic data into a mathematically clean dataset ready for machine learning algorithms. I handled missing values using global median imputation and conditional subgrouping instead of arbitrary guesswork, and dropped features with over 20% missing data. I then neutralized extreme outliers by establishing mathematical boundaries using the Interquartile Range (IQR). Finally, I engineered three new predictive features: `Release_Year` (temporal extraction), `Engagement_Score` (optimization target), and `Genre_Count` (complexity index).

### **Project 2: Supervised Learning & Fraud Detection**
* **Dataset Used:** `loan_data_new.csv` – A highly imbalanced financial dataset containing loan details and a 'Loan Status' target variable used for default/fraud detection.
* **What I Did:** I built a mathematically secure classification pipeline utilizing the Random Forest algorithm. To prevent data leakage, I implemented a strict "Zero-Leakage Protocol" using `imblearn.pipeline`, ensuring that SMOTE (Synthetic Minority Over-sampling) was only applied to the training folds during cross-validation. I holistically tuned the hyperparameters using `GridSearchCV` and evaluated the model's performance on an untouched test set using strict Precision, Recall, and ROC-AUC metrics instead of misleading accuracy scores.

### **Project 3: Unsupervised Learning & Dimensionality Reduction**
* **Dataset Used:** `world_population.csv` – A global demographic dataset containing 12+ numeric features like historical populations, density, and growth rates for various countries.
* **What I Did:** I transitioned to discovering hidden mathematical groupings in unlabeled data. I first standardized the numeric features and applied Principal Component Analysis (PCA) to compress the high-dimensional data into 3 dimensions while retaining core behavioral signals. I then mathematically proved the optimal number of clusters by utilizing diagnostic gatekeepers—specifically the Elbow Method (to minimize variance) and the Silhouette Score (to maximize separation). Finally, I deployed a K-Means algorithm (with K=4) to translate these spatial coordinates into actionable, human-centric "Demographic Personas".

## **To run and test the code:**

**1.** Open Google Colab or your local Jupyter Notebook environment.

**2.** Each project notebook requires a specific dataset to run successfully. Ensure the correct CSV file is placed in the exact same working directory as the notebook. If you are using Google Colab, click the folder icon on the left sidebar and upload the datasets to the session storage.
* **For Week 1 (`Decode Labs Week 1.ipynb`):** Upload `mymoviedb.csv`
* **For Week 2 (`Decode Labs Week 2.ipynb`):** Upload `loan_data_new.csv`
* **For Week 3 (`Decode Labs Week 3.ipynb`):** Upload `world_population.csv`

**3.** If using Colab, you can upload the dataset directly to the session storage via the folder icon on the left sidebar.

**4.** Open the desired notebook (`.ipynb` file). Read through the Markdown documentation, and run the code cells sequentially (Shift + Enter) or simply select **"Run All"** from the top menu. This will execute the pipelines, output the cleaned data structures, and display the final model metrics.
