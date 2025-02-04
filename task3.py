# Reimportar librerías después del reset del entorno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset nuevamente
file_path = "high_diamond_ranked_10min.csv"
df = pd.read_csv(file_path)

# Eliminar columna irrelevante
df = df.drop(columns=['gameId'])

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=['blueWins'])
y = df['blueWins']

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dataset en 80% entrenamiento, 10% validación y 10% prueba
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Extraer solo las dos características clave
feature_1_name = "blueTotalGold"
feature_2_name = "blueExperienceDiff"

# Obtener los índices de estas características en el dataset original
feature_1_idx = X.columns.get_loc(feature_1_name)
feature_2_idx = X.columns.get_loc(feature_2_name)

# Seleccionar solo estas características en los conjuntos de datos
X_train_selected = X_train[:, [feature_1_idx, feature_2_idx]]
X_test_selected = X_test[:, [feature_1_idx, feature_2_idx]]

# Implementación del modelo SVM sin librerías
class SupportVectorMachineOptimized:
    def __init__(self, learning_rate=0.001, lambda_param=0.005, n_iters=1500):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 0, -1, 1)  # Convertir 0 a -1

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            condition = y_ * (np.dot(X, self.w) + self.b) >= 1
            dw = 2 * self.lambda_param * self.w - np.dot(X.T, y_ * ~condition)
            db = -np.sum(y_ * ~condition)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)  # -1 o 1

# Entrenar un nuevo modelo con solo estas dos características clave
svm_selected = SupportVectorMachineOptimized(learning_rate=0.001, lambda_param=0.005, n_iters=1500)
svm_selected.fit(X_train_selected, y_train)

# Evaluar el modelo con las características seleccionadas
y_pred_selected = svm_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, np.where(y_pred_selected == -1, 0, 1))
report_selected = classification_report(y_test, np.where(y_pred_selected == -1, 0, 1))

# Calcular métricas adicionales
precision = np.sum((y_test == 1) & (y_pred_selected == 1)) / np.sum(y_pred_selected == 1)
recall = np.sum((y_test == 1) & (y_pred_selected == 1)) / np.sum(y_test == 1)
f1_score = 2 * (precision * recall) / (precision + recall)

# Calcular tasa de falsos positivos y falsos negativos
false_positive_rate = np.sum((y_test == 0) & (y_pred_selected == 1)) / np.sum(y_test == 0)
false_negative_rate = np.sum((y_test == 1) & (y_pred_selected == 0)) / np.sum(y_test == 1)

# Número total de predicciones correctas e incorrectas
correct_predictions = np.sum(y_test == y_pred_selected)
incorrect_predictions = np.sum(y_test != y_pred_selected)

# Mostrar las estadísticas
stats = {
    "Accuracy": accuracy_selected,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score,
    "False Positive Rate": false_positive_rate,
    "False Negative Rate": false_negative_rate,
    "Correct Predictions": correct_predictions,
    "Incorrect Predictions": incorrect_predictions,
    "Total Samples": len(y_test)
}

import pandas as pd

# Convertir el diccionario de estadísticas en un DataFrame para mostrarlo
df_stats = pd.DataFrame(stats.items(), columns=["Métrica", "Valor"])
print("Estadísticas del modelo SVM")
print(df_stats)

# Regenerar la malla de puntos para la visualización
x_min, x_max = X_test_selected[:, 0].min() - 1, X_test_selected[:, 0].max() + 1
y_min, y_max = X_test_selected[:, 1].min() - 1, X_test_selected[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm_selected.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la nueva frontera de decisión con las características seleccionadas
plt.figure(figsize=(8, 6))
plt.scatter(X_test_selected[y_test == 0][:, 0], X_test_selected[y_test == 0][:, 1], color='red', label='Perdió (0)', alpha=0.5)
plt.scatter(X_test_selected[y_test == 1][:, 0], X_test_selected[y_test == 1][:, 1], color='blue', label='Ganó (1)', alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

plt.xlabel(feature_1_name + ' (Normalizado)')
plt.ylabel(feature_2_name + ' (Normalizado)')
plt.title('Frontera de Decisión del SVM (Sin Librerías)')
plt.legend()
plt.show()
