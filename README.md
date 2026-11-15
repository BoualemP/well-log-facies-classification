# Well-Log Facies Classification with a Neural Network

Prediction of nine lithofacies from wireline logs in the SEG 2016 machine learning contest dataset. A Keras neural network is compared with logistic regression and gradient boosting, every model is evaluated on wells held out entirely from training, and the network is exported to ONNX for a web application that classifies a new well from a CSV file.

This project continues [Well-Log Machine Learning from Scratch](https://github.com/BoualemP/well-log-ml-from-scratch), which separated marine from non-marine intervals in the same dataset with linear models. It moves to the full nine-facies problem and follows that project's next steps: depth-context features and non-linear models.

## Results

Macro F1 (every facies weighs the same); test wells NEWBY and SHANKLE:

| Model | Cross-validation by well (mean, range) | Test wells | Test accuracy |
|---|---|---|---|
| Majority class | 0.049 (0.03–0.06) | 0.038 | 0.205 |
| Logistic regression | 0.317 (0.29–0.35) | 0.399 | 0.509 |
| Gradient boosting | 0.338 (0.27–0.47) | 0.396 | 0.493 |
| **Neural network** | 0.312 (0.29–0.35) | **0.446** | 0.476 |

## Key findings

- The three trained models score six to seven times the benchmark, but cross-validation does not separate them: their differences are smaller than the variation between wells.
- The neural network has the best macro F1 on the test wells; logistic regression has the best accuracy.
- Most errors fall between facies with adjacent codes (CSiS / FSiS, WS / PS); SS is rarely recognised, as almost all its training samples come from one well.
- The network is selected for the web application: it converts to a 15.5 kB ONNX file whose probabilities match the Keras model (largest difference 3e-7).

## Methodology

- Raw data inspection: missing PE logs, an incomplete well, facies counts per well and extreme values; logs capped at training percentiles.
- Depth context: gradient and 5-sample moving average of each log within each well.
- Tuning of each model by leave-one-well-out cross-validation on five training wells.
- Final evaluation on two wells never used for tuning: macro F1, accuracy, confusion matrix, log displays and fragmentation of the predicted facies columns.
- Export to ONNX with the feature settings (caps, standardisation, facies codes) and two demonstration wells.

## Tech stack

Python, TensorFlow / Keras, scikit-learn, ONNX (tf2onnx, onnxruntime), pandas, NumPy, matplotlib, seaborn, Jupyter.

## Project structure

```
well-log-facies-classification/
├── well_log_facies_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Boualem Pereira**
