# GuardAI: A Hybrid IDS using Decision Trees and Random Forests

GuardAI is an Intrusion Detection System (IDS) that leverages the combined strengths of Decision Trees and Random Forests to detect malicious activities in network traffic with higher accuracy and robustness. This hybrid approach integrates interpretable decision paths with the ensemble learning power of random forests to provide both explainability and performance in intrusion detection.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Architecture](#architecture)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Future Improvements](#future-improvements)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

Intrusion Detection Systems are crucial for identifying and mitigating cyber threats in real time.
While **Decision Trees** provide clear and explainable classification rules, they can overfit.
**Random Forests** mitigate this by using multiple trees to achieve better generalization.

GuardAI combines these two approaches:

* Decision Trees for **rule-based explainability** (security teams can understand the "why").
* Random Forests for **robust detection** even in noisy or imbalanced datasets.

---

## Features

* Hybrid classification model using Decision Trees + Random Forests
* Supports binary and multi-class intrusion detection.
* Preprocessing pipeline for cleaning and encoding network traffic data.
* Handles imbalanced datasets using SMOTE or class weighting.
* Generates performance metrics: accuracy, precision, recall, F1-score, confusion matrix.
* Easy integration with other security monitoring tools.
* Lightweight, Python-based, and easily deployable.

---

## **Tech Stack**

* Programming Language: Python 3.x
* Libraries:

  * `pandas` – Data handling
  * `numpy` – Numerical operations
  * `scikit-learn` – ML algorithms & metrics
  * `matplotlib` & `seaborn` – Data visualization
  * `imblearn` – SMOTE for class balancing
  * Jupyter Notebook – Model development & testing

---

## Architecture

```
 ┌─────────────────────┐
 │  Dataset Loading     │
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │ Data Preprocessing  │
 │ - Cleaning          │
 │ - Encoding          │
 │ - Scaling           │
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │ Model Training      │
 │ - Decision Tree     │
 │ - Random Forest     │
 │ - Hybrid Strategy   │
 └─────────┬───────────┘
           │
 ┌─────────▼───────────┐
 │ Evaluation & Output │
 │ - Metrics           │
 │ - Confusion Matrix  │
 │ - Feature Importance│
 └─────────────────────┘
```

---

## Dataset

* Recommended Dataset:[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) or [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
* Dataset preprocessing steps:

  * Remove duplicate entries.
  * Encode categorical features.
  * Normalize numerical features.
  * Handle imbalanced classes.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/GuardAI-A-Hybrid-IDS-using-Decision-Trees-and-Random-Forests.git
   cd GuardAI-A-Hybrid-IDS-using-Decision-Trees-and-Random-Forests
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Prepare the dataset and place it in the `data/` directory.
2. **Run the training script**:

   ```bash
   python train.py --dataset data/nsl_kdd.csv
   ```
3. Evaluate the model:

   ```bash
   python evaluate.py --model hybrid_model.pkl --test data/test.csv
   ```
4. View results** (Confusion Matrix, Feature Importance):

   * Plots will be saved in the `results/` directory.

---

## Results

Example performance (on NSL-KDD dataset):

| Model              | Accuracy  | Precision | Recall    | F1-Score  |
| ------------------ | --------- | --------- | --------- | --------- |
| Decision Tree      | 91.2%     | 90.5%     | 90.1%     | 90.3%     |
| Random Forest      | 95.6%     | 95.2%     | 95.0%     | 95.1%     |
| Hybrid GuardAI     | 96.4%     | 96.1%     | 96.0%     | 96.0%     |

---

## Future Improvements

* Real-time traffic analysis with packet capture.
* Integration with SIEM tools (e.g., Splunk, ELK Stack).
* Deep learning extension (e.g., LSTMs for sequential network data).
* Web-based dashboard for live monitoring.

---

## Contributing

Contributions are welcome!

1. Fork this repository.
2. Create a new branch:

   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:

   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to your branch and submit a Pull Request.

---
