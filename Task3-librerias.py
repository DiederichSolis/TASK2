import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#  Cargar el dataset
file_path = "high_diamond_ranked_10min.csv"
df = pd.read_csv(file_path)

#  Eliminar columna irrelevante
df = df.drop(columns=['gameId'])

#  Seleccionar características y variable objetivo
X = df.drop(columns=['blueWins'])
y = df['blueWins']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Dividir dataset en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Definir el modelo SVM con hiperparámetros ajustados
svm_model = SVC(kernel="rbf", C=10, gamma=0.3)
svm_model.fit(X_train, y_train)

#  Evaluar el modelo
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n Accuracy: {accuracy:.4f}")
print("\n Classification Report:")
print(report)

#  Seleccionar las características para visualización
feature_1 = "blueTotalGold"
feature_2 = "blueExperienceDiff"

X_visual = df[[feature_1, feature_2]]
y_visual = df["blueWins"]

#  Normalizar las dos características seleccionadas
X_visual_scaled = scaler.fit_transform(X_visual)

#  Entrenar SVM con solo dos características
svm_visual = SVC(kernel="rbf", C=10, gamma=0.3)
svm_visual.fit(X_visual_scaled, y_visual)

#  Generar una malla de puntos
xx, yy = np.meshgrid(np.linspace(X_visual_scaled[:, 0].min(), X_visual_scaled[:, 0].max(), 100),
                     np.linspace(X_visual_scaled[:, 1].min(), X_visual_scaled[:, 1].max(), 100))
Z = svm_visual.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#  Graficar la frontera de decisión
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
sns.scatterplot(x=X_visual_scaled[:, 0], y=X_visual_scaled[:, 1], hue=y_visual, palette="coolwarm", edgecolor="k")
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Frontera de Decisión con SVM (scikit-learn)")
plt.show()
