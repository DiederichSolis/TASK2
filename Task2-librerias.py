import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Lectura y preparación de los datos desde el archivo TXT
# Se asume que cada línea tiene la estructura:
#   etiqueta mensaje...
# Ejemplo:
#   ham Esto es un mensaje de ejemplo.
#   spam ¡Gana dinero rápido sin esfuerzo!
data = []
archivo = "entrenamiento.txt"  # Asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta completa

with open(archivo, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue  # Se saltan las líneas vacías
        try:
            label, message = line.split(" ", 1)
        except ValueError:
            # Si la línea no sigue el formato esperado, se ignora
            continue
        data.append({"label": label, "message": message})

# Convertir la lista de diccionarios en un DataFrame
df = pd.DataFrame(data)
print("Datos de ejemplo:")
print(df.head())

# 2. Preparación de los datos para entrenamiento (se usan todos los datos)
X = df['message']
y = df['label']

# 3. Transformación del texto a vectores numéricos
# Se utiliza CountVectorizer para convertir cada mensaje en un vector de frecuencias
vectorizer = CountVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

# 4. Entrenamiento del clasificador Naive Bayes con Laplace smoothing
# El parámetro alpha=1.0 aplica Laplace smoothing
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_vect, y)

# 5. Evaluación del modelo sobre los datos de entrenamiento (una única impresión)
y_pred = classifier.predict(X_vect)
accuracy = accuracy_score(y, y_pred)
print("\n--- Evaluación del modelo (sobre datos de entrenamiento) ---")
print("Precisión del modelo:", accuracy)
print("\nMatriz de confusión:")
print(confusion_matrix(y, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y, y_pred))

# 6. Clasificación de un mensaje ingresado por el usuario
mensaje_usuario = input("\nIngrese un mensaje para clasificarlo (Spam o Ham): ")
mensaje_vect = vectorizer.transform([mensaje_usuario])
prediccion = classifier.predict(mensaje_vect)
print("\nEl mensaje ingresado es clasificado como:", prediccion[0])

