# Parkinson's Detection

This repository contains a machine learning project aimed at detecting Parkinson's Disease using a dataset of medical voice measurements. The project includes a Jupyter Notebook that preprocesses the data, trains a machine learning model, and evaluates its performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Parkinson's Disease is a progressive nervous system disorder that affects movement. Early detection can help improve patient outcomes. This project leverages machine learning techniques to analyze vocal features and predict the presence of Parkinson's Disease.

## Dataset

The dataset used in this project is included in the file `parkinsons.csv`. It contains the following key features:

- **MDVP:Fo(Hz)**: Average vocal fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
- Various other measures of vocal signal, including jitter, shimmer, and harmonic-to-noise ratio
- **status**: The target variable, where 1 indicates Parkinson's Disease and 0 indicates no disease

### Source
The dataset is publicly available and has been used widely for Parkinson's Disease detection research.

## Dependencies

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using the command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/parkinsons-detection.git
   cd parkinsons-detection
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook "Parkinson's_Detection.ipynb"
   ```

3. Run the cells in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

## Project Workflow

1. **Data Loading**: Load the dataset and inspect its structure.
2. **Exploratory Data Analysis (EDA)**: Visualize the data and analyze feature distributions.
3. **Data Preprocessing**:
   - Handle missing values (if any).
   - Normalize or standardize the features.
4. **Model Training**:
   - Train a machine learning model (e.g., Logistic Regression, SVM, Random Forest).
   - Perform hyperparameter tuning to optimize model performance.
5. **Evaluation**:
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
   - Plot confusion matrix and other relevant visualizations.

## Results

The trained model achieved the following performance:

- **Accuracy**: _Add Accuracy_
- **Precision**: _Add Precision_
- **Recall**: _Add Recall_
- **F1-Score**: _Add F1-Score_

Further details and visualizations are included in the notebook.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

