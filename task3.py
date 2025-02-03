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

# Dividir dataset en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Implementación del modelo SVM sin librerías
class SupportVectorMachineOptimized:
    def __init__(self, learning_rate=0.0001, lambda_param=0.00001, n_iters=1000):
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

# Entrenar modelo
svm = SupportVectorMachineOptimized(learning_rate=0.0059, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Evaluación del modelo
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, np.where(y_pred == -1, 0, 1))
report = classification_report(y_test, np.where(y_pred == -1, 0, 1))

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Visualización de frontera de decisión con 2 características
feature_1 = 0  # Primera característica
feature_2 = 1  # Segunda característica

X_train_vis = X_train[:, [feature_1, feature_2]]
X_test_vis = X_test[:, [feature_1, feature_2]]

# Entrenar un nuevo modelo con solo dos características
svm_vis = SupportVectorMachineOptimized(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm_vis.fit(X_train_vis, y_train)

# Generar una malla de puntos
x_min, x_max = X_test_vis[:, 0].min() - 1, X_test_vis[:, 0].max() + 1
y_min, y_max = X_test_vis[:, 1].min() - 1, X_test_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar frontera de decisión
plt.figure(figsize=(8, 6))
plt.scatter(X_test_vis[y_test == 0][:, 0], X_test_vis[y_test == 0][:, 1], color='red', label='Perdió (0)', alpha=0.5)
plt.scatter(X_test_vis[y_test == 1][:, 0], X_test_vis[y_test == 1][:, 1], color='blue', label='Ganó (1)', alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

plt.xlabel('Característica 1 (Normalizada)')
plt.ylabel('Característica 2 (Normalizada)')
plt.title('Frontera de Decisión del SVM (Sin Librerías)')
plt.legend()
plt.show()
