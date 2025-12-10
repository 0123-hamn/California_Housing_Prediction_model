
---

# 🏡 California Housing Price Prediction

This repository demonstrates a full machine-learning workflow for predicting **median house values** in California districts using the classic California Housing dataset.

It includes:

* Exploratory Data Analysis (EDA) in Jupyter Notebook
* Data preprocessing pipelines
* Model training using Random Forest
* Saving/loading ML models using Joblib
* Running inference from CSV input
* Auto-generated prediction 
* Deployed at streamlit
* <img width="1602" height="918" alt="image" src="https://github.com/user-attachments/assets/510d7e11-9a52-4051-8181-5acec9d7d2ef" />

  

---

## 📁 Project Structure

```
├── California_Housing.ipynb   # Jupyter notebook for EDA & data exploration
├── housing.csv                # Original dataset used for training
├── input.csv                  # Input file for generating predictions
├── output.csv                 # Output file with predicted values
├── main.py                    # Main script for training & inference
├── model.pkl                  # Saved Random Forest model (created after training)
├── pipeline.pkl               # Saved preprocessing pipeline
└── README.md                  # Project documentation
```

---

## 🚀 Features

### ✔ Automated ML Pipeline

`main.py` builds a preprocessing pipeline using:

* **SimpleImputer** → handles missing data
* **StandardScaler** → scales numerical columns
* **OneHotEncoder** → encodes categorical features

### ✔ Model: Random Forest Regressor

A robust regression algorithm from scikit-learn.

### ✔ Smart Train/Inference Mode

The script automatically detects whether a model already exists:

* If `model.pkl` **does not exist** → Train a new model
* If `model.pkl` **exists** → Perform inference on `input.csv`

### ✔ Stratified Sampling

Stratified shuffle split based on income category ensures balanced, unbiased sampling.

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <repository-folder>
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, use:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## 🏋️‍♂️ Training the Model

To train the model (runs automatically if `model.pkl` is missing):

```bash
python main.py
```

This will:

1. Load `housing.csv`
2. Build preprocessing pipeline
3. Train Random Forest
4. Save:

   * `model.pkl`
   * `pipeline.pkl`

---

## 🔮 Running Inference

Place your input data in `input.csv`, then run:

```bash
python main.py
```

The script will:

* Load the trained model & pipeline
* Transform input data
* Predict median house value
* Save predictions to:

```
output.csv
```

---

## 📊 Example Output (`output.csv`)

| longitude | latitude | housing_median_age | ... | median_house_value |
| --------- | -------- | ------------------ | --- | ------------------ |
| -122.23   | 37.88    | 41                 | ... | 265432.12          |
| -118.31   | 34.02    | 28                 | ... | 401221.55          |

---

## 📓 EDA Notebook

The file **California_Housing.ipynb** contains:

* Data visualization
* Correlation analysis
* Distribution & outlier checks
* Feature engineering exploration

This notebook helps you understand the dataset before training.

---

## 🛠 Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## 🤝 Contributing

Pull requests and suggestions are welcome!
If you find a bug or want a new feature, feel free to open an issue.

---

## 📄 License

This project is open-source under the **MIT License**.

---

