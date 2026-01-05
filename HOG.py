import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

IMG_SIZE = 224
DATA_DIR = 'dataset_divididoybalaceado'

# Descriptores HOG 
hog = cv2.HOGDescriptor(
    _winSize=(224, 224),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9
)

# Cargamos y procesamos los datos con Hog
def load_data_hog(split_name):
    path = os.path.join(DATA_DIR, split_name)
    features, labels = [], []

    if not os.path.exists(path):
        return np.array([]), np.array([]), []

    categories = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    for idx, category in enumerate(categories):
        category_path = os.path.join(path, category)
        print(f"Procesando {split_name} - Clase: {category}")

        for img_name in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            hog_features = hog.compute(img).flatten()

            features.append(hog_features)
            labels.append(idx)

    return np.array(features), np.array(labels), categories

# Matriz de confusión
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - Modelo Final')
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción del Modelo')
    plt.show()


if __name__ == '__main__':

    #Seleccionamos los datos ya divididos
    X_train, y_train, classes = load_data_hog('train')
    X_test, y_test, _ = load_data_hog('test')

    print(f"\nDimensión del vector HOG: {X_train.shape[1]}")

    # -----------------------------
    # 2. Análisis de sobreajuste
    # -----------------------------
    depth_range = [8, 10, 12, 14, 16, None]
    train_scores = []
    test_scores = []

    print("\nAnalizando el sobreajuste variando max_depth...")

    best_acc = 0
    best_depth = None

    for depth in depth_range:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=depth,
            min_samples_leaf=3,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        rf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, rf.predict(X_test))

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        print(f"Profundidad: {depth} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_depth = depth
            joblib.dump(rf, 'modelo_hog_random_forest.joblib')

    # -----------------------------
    # 3. Gráfica
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(depth_range, train_scores, 'o--', label='Train')
    plt.plot(depth_range, test_scores, 's-', label='Test')
    plt.title('HOG + RF: Análisis de Sobreajuste')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # 4. Evaluación final
    # -----------------------------
    print(f"\nCargando mejor modelo (max_depth={best_depth})...")
    final_model = joblib.load('modelo_hog_random_forest.joblib')

    y_pred = final_model.predict(X_test)
    print(f"Accuracy final en TEST: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=classes))

    plot_confusion_matrix(y_test, y_pred, classes)
