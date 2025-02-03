import numpy as np
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Task 2.1 - Lectura y limpieza del dataset
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    messages, labels = [], []
    for line in lines:
        parts = line.strip().split('\t') 
        if len(parts) == 2:
            label = parts[0].lower().strip()
            message = parts[1].strip()
            labels.append(label)
            messages.append(message)
    return messages, labels


dataset_file = "entrenamiento.txt"
X, y = load_data(dataset_file)


print("Distribución de clases en el dataset:")
print({label: y.count(label) for label in set(y)})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
