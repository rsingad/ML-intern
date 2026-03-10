# Restaurant Data Analysis & Machine Learning Project

This repository contains the Machine Learning projects developed during my internship at **Cognifyz Technologies**. The project focuses on analyzing restaurant data to predict ratings, classify cuisines, and building a recommendation system.

## 🚀 Key Features

### 1. Restaurant Rating Prediction
- Predicts the aggregate rating of a restaurant based on various features.
- **Algorithms:** Linear Regression, Decision Tree Regressor.
- **Performance:** Achieved an R-squared score of **0.92** with Decision Tree.

### 2. Cuisine Classification
- Classifies restaurants based on their cuisines (Top 20 most frequent).
- **Algorithm:** Random Forest Classifier.
- **Evaluation:** Includes detailed classification reports and confusion matrices.

### 3. Restaurant Recommendation System
- Provides personalized restaurant suggestions using content-based filtering.
- **Techniques:** TF-IDF Vectorization, Cosine Similarity.
- **Functions:** Recommendation by restaurant similarity or by user preference (Cuisine & Price).

### 4. Location-based Analysis
- Geographical distribution visualization using Latitude and Longitude.
- Statistical analysis of ratings and prices across different cities.
- Identification of city-wise trends and patterns.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Documentation:** Markdown

## 🖥️ How to Run

1. **Clone/Download** the repository.
2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. **Run the Project:**
   ```bash
   python3 main.py
   ```
   *Follow the interactive menu to run specific tasks.*

## 📂 Project Structure
- `predict_ratings.py`: Rating prediction logic.
- `classify_cuisines.py`: Cuisine classification logic.
- `recommend_restaurants.py`: Recommendation engine.
- `location_analysis.py`: Geographical analysis script.
- `main.py`: Central entry point for all tasks.
- `Dataset (1).csv`: The dataset used for analysis.
- `beginner_guide.md`: Simplified project explanation.

## 🤝 Acknowledgements
Special thanks to **Cognifyz Technologies** for the internship opportunity and guidance.

---
**Hashtags:** #cognifyztechnologies #cognifyz #cognifyztech #machinelearning #internship
# ML-intern
