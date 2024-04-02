import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error



def main():
    get_best_knn_classificator()
    get_best_knn_regressor()


def get_best_knn_classificator():
    print("KNN-класифікатор:")

    np.random.seed(2021)

    # Завантаження бази даних
    iris = load_iris()
    X, y, labels, feature_names = iris.data, iris.target, iris.target_names, iris['feature_names']

    # Перемішування записів
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Нормалізація параметрів
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Розділення на навчальну і тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Навчання KNN-класифікатора з різними значеннями K та вибір найкращого K
    k_best = None
    score_best = 0

    for k in range(1, 21):  # Перебираємо значення K від 1 до 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        if score > score_best:
            score_best = score
            k_best = k

    print('The best k = {} , score = {}'.format(k_best, score_best))


def get_best_knn_regressor():
    print("KNN-регресор:")

    # Згенерувати випадковий набір даних
    np.random.seed(42)
    X = np.random.rand(1000, 1) * 100  # 1000 значень у діапазоні [0, 100]
    y = 3 * X.squeeze() + np.random.randn(1000) * 10  # шум

    # Нормалізувати дані
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Розділити дані на навчальний і тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Навчити KNN-регресор з різними значеннями K
    ks = range(1, 21)
    mse_values = []

    for k in ks:
        knn_regressor = KNeighborsRegressor(n_neighbors=k)
        knn_regressor.fit(X_train, y_train)
        y_pred = knn_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    # Знайти значення K з найменшим середнім квадратичним відхиленням
    best_k = ks[np.argmin(mse_values)]

    # Візуалізувати результати
    plt.plot(ks, mse_values, marker='o')
    plt.title('Залежність середньоквадратичної помилки від K')
    plt.xlabel('K')
    plt.ylabel('Середня квадратична помилка')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

    print(f"Найкраще значення K: {best_k}")


if __name__=="__main__":
    main()