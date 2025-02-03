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


# Task 2.2 - Construcción del modelo
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace Smoothing
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()
    
    def preprocess(self, text):
        text = text.lower()  
        text = re.sub(r'[^a-z0-9\s]', '', text)  
        return text.split()
    
    def fit(self, X_train, y_train):
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_docs = len(y_train)
        
        for text, label in zip(X_train, y_train):
            class_counts[label] += 1
            words = self.preprocess(text)
            self.vocab.update(words)
            for word in words:
                word_counts[label][word] += 1
        
        self.class_probs = {label: count / total_docs for label, count in class_counts.items()}
        
        self.word_probs = {}
        for label, words in word_counts.items():
            total_words = sum(words.values()) + self.alpha * len(self.vocab)
            self.word_probs[label] = {
                word: (count + self.alpha) / total_words
                for word, count in words.items()
            }
    
    def predict(self, X_test):
        predictions = []
        for text in X_test:
            words = self.preprocess(text)
            scores = {}
            for label in self.class_probs:
                score = np.log(self.class_probs[label])
                for word in words:
                    if word in self.word_probs[label]:
                        score += np.log(self.word_probs[label][word])
                    else:
                        score += np.log(self.alpha / (sum(self.word_probs[label].values()) + self.alpha * len(self.vocab)))
                scores[label] = score
            best_label = max(scores, key=scores.get)
            predictions.append(best_label)
        return predictions


nb_classifier = NaiveBayesClassifier(alpha=1.0)
nb_classifier.fit(X_train, y_train)


y_pred = nb_classifier.predict(X_test)
print("\n=== Evaluación del modelo ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='spam', zero_division=1))
print("Recall:", recall_score(y_test, y_pred, pos_label='spam', zero_division=1))
print("F1-score:", f1_score(y_test, y_pred, pos_label='spam', zero_division=1))


conf_matrix = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
