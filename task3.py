import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset
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

# Generar la malla de puntos para la visualización
x_min, x_max = X_test_selected[:, 0].min() - 1, X_test_selected[:, 0].max() + 1
y_min, y_max = X_test_selected[:, 1].min() - 1, X_test_selected[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm_selected.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("Precisión del Modelo (Características Seleccionadas):", accuracy_selected)

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

# Retornar los resultados de precisión
accuracy_selected, report_selected
