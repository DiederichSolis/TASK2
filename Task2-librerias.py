import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score 

def normalize_text(text):
    """
    Normaliza el texto convirtiéndolo a minúsculas, eliminando puntuación, dígitos, URLs y espacios extra.
    """
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Eliminar URLs
    text = re.sub(r'\S+@\S+', '', text)  # Eliminar correos electrónicos
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación y caracteres especiales
    text = re.sub(r'\d+', '', text)  # Eliminar dígitos
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    return text

data = []
archivo = "entrenamiento.txt"

with open(archivo, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        try:
            label, message = line.split("\t", 1)
        except ValueError:
            continue
        data.append({"label": label, "message": message})


df = pd.DataFrame(data)
print("Datos de ejemplo:")
print(df.head())

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english', preprocessor=normalize_text)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_train_vect, y_train)

y_pred = classifier.predict(X_test_vect)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")


print("\n--- Evaluación del modelo (sobre datos de prueba) ---")
print("Precisión (Accuracy):", accuracy)
print("Precisión (Precision):", precision)
print("Recall (Sensibilidad):", recall)
print("F1-Score:", f1)
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

mensaje_usuario = input("\nIngrese un mensaje para clasificarlo (Spam o Ham): ")
mensaje_vect = vectorizer.transform([mensaje_usuario])
prediccion = classifier.predict(mensaje_vect)
print("\nEl mensaje ingresado es clasificado como:", prediccion[0])
