# Cancer Rate Prediction using Deep Learning

## Project Description
This project focuses on applying deep learning techniques to predict cancer rates (high or low) based on socio-economic and environmental factors across various countries. The model uses a combination of neural network architectures and hyperparameter tuning to improve prediction accuracy.

## Project Structure
- `data/`: Contains the dataset `CategoricalCancerRates.csv` used for training and testing.
- `notebooks/`: Jupyter notebooks for model training, tuning, and evaluation.
- `models/`: Saved models for best-performing architectures.
- `results/`: Output results, evaluation metrics, and graphs from model training.
  
## Dataset
The dataset consists of various socio-economic and environmental indicators such as GDP per capita, urban population, CO2 emissions, life expectancy, and other health-related features for different countries. The target variable is `CANRAT`, which classifies cancer rates as "High" or "Low".

## Steps for Implementation

1. **Data Preprocessing:**
   - Missing values handled using median/mode imputation.
   - Categorical variables like `CANRAT` and `HDICAT` were label-encoded.
   
2. **Modeling:**
   - Multiple deep learning models, including baseline models, deeper networks, models with dropout layers, and L2 regularization, were implemented.
   - Hyperparameters such as the number of neurons, learning rate, batch size, and epochs were optimized using grid search.

3. **Evaluation:**
   - Models were evaluated using accuracy, classification report, and confusion matrix.
   - Training and validation accuracy/loss were visualized to monitor performance over epochs.

4. **Tuning:**
   - Hyperparameter tuning using grid search was conducted to find the best combination of batch size, number of neurons, learning rate, and dropout rates for optimal performance.

## Key Results
- The best-performing model achieved an accuracy of **X%** on the test dataset after extensive hyperparameter tuning.
- Dropout layers and regularization were effective in mitigating overfitting and improving generalization on the test data.
  
## Requirements
- Python 3.x
- Libraries:
  - TensorFlow/Keras
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - Scikit-learn

You can install the required dependencies using the following command:

```bash


