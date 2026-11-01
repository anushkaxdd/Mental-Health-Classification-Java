# Mental Health BERT Fine-Tuned – Kaggle Evaluation Notebook

## Overview

This project evaluates a fine-tuned BERT model for mental health text
classification and implements model inference and evaluation in Java
using ONNX Runtime.

The system classifies text into four categories:

- Anxiety
- Depression
- Normal
- Suicidal

The project demonstrates NLP model deployment, Java-based inference,
evaluation metrics, model explainability and cross-framework validation.

## Dataset

Source:
Kaggle - Mental Health Text Classification Dataset

Test dataset:
992 samples

Classes:
- Anxiety: 248
- Depression: 248
- Normal: 248
- Suicidal: 248

## Model

Architecture: BERT

Vocabulary size: 30,522

Hidden size: 768

Transformer layers: 12

Attention heads: 12

Maximum sequence length: 128

Number of output classes: 4

## Technology Stack

- Java 17+
- Python (one-time ONNX conversion and analysis)
- ONNX Runtime
- PyTorch
- Hugging Face Transformers
- WordPiece Tokenization
- OpenCSV
- Maven
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- LIME

## System Architecture

1. Load the fine-tuned PyTorch BERT checkpoint.
2. Export the model into ONNX format.
3. Load the ONNX model using Java ONNX Runtime.
4. Tokenize input text using WordPiece.
5. Generate predictions and class probabilities.
6. Evaluate predictions against the test dataset.
7. Generate performance metrics and visualizations.
8. Compare original PyTorch and ONNX outputs.

Python is used for one-time model conversion and supplementary
evaluation visualizations. Java handles the deployed inference
and primary evaluation pipeline.

## Java Evaluation Results

Test samples: 992

Accuracy: 89.72%

Macro Precision: 89.56%

Macro Recall: 89.72%

Macro F1-score: 89.54%

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Anxiety | 0.8760 | 0.8548 | 0.8653 |
| Depression | 0.8616 | 0.7782 | 0.8178 |
| Normal | 0.9380 | 0.9758 | 0.9565 |
| Suicidal | 0.9067 | 0.9798 | 0.9419 |

## Additional Evaluation

| Metric | Result |
|---|---:|
| Macro ROC-AUC (OVR) | 0.9771 |
| Macro PR-AUC | 0.9435 |
| Top-label ECE | 0.0871 |
| Multiclass Brier Score | 0.1864 |
| Original NLL | 0.5573 |
| Temperature-scaled NLL | 0.3418 |

Optimal exploratory temperature: 2.2352.

Threshold sensitivity analysis and LIME explainability
were also performed.

Note: Temperature scaling was fitted using test-set labels.
Its results are exploratory and should not be interpreted
as unbiased calibration performance.

## PyTorch vs ONNX Validation

The original PyTorch model and ONNX model were evaluated
on all 992 test samples.

- Matching predictions: 992
- Mismatching predictions: 0
- Prediction agreement: 100%
- Maximum absolute logit difference: 0.0000161
- PyTorch accuracy: 89.72%
- ONNX accuracy: 89.72%

This demonstrates numerical and prediction-level agreement
between the original model and its ONNX export on the test dataset.

## Project Structure

MentalHealthBERT/
|
|-- models/
|   |-- model.onnx
|   |-- config.json
|   |-- vocab.txt
|   |-- test.csv
|
|-- src/
|   |-- main/
|       |-- java/
|           |-- Main.java
|           |-- Evaluation.java
|           |-- Tokenizer.java
|           |-- TokenizerTest.java
|
|-- results/
|   |-- model_predictions.csv
|   |-- classification_report.txt
|   |-- confusion_matrix.csv
|   |-- misclassified_examples.csv
|   |-- confusion_matrix.png
|   |-- classwise_performance.png
|   |-- class_distribution.png
|   |-- confidence_distribution.png
|   |-- roc_auc_curves.png
|   |-- precision_recall_curves.png
|   |-- calibration_diagram.png
|   |-- temperature_scaling.png
|   |-- threshold_sensitivity.png
|   |-- lime_explanation.html
|   |-- pytorch_onnx_comparison.csv
|   |-- full_model_validation.csv
|
|-- pom.xml
|-- README.md

## Running the Java Project

Requirements:
- Java 17 or later
- Maven

Compile:

mvn clean compile

Run single-text inference:

mvn exec:java -Dexec.mainClass=Main

Run complete evaluation:

mvn clean compile exec:java -Dexec.mainClass=Evaluation

## Limitations

- This is an educational and research-oriented NLP project.
- Classification results depend on the training dataset and its biases.
- Predictions should not be interpreted as clinical diagnoses.
- The model is not a suicide-risk assessment tool.
- LIME explanations are approximate and local.
- Calibration results require independent validation data.
- Performance on other datasets and real-world populations
  has not been established.

## Responsible Use

This model must not be used as a replacement for professional
mental health assessment, diagnosis or emergency support.
