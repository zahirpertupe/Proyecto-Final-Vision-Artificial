import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- CARGA DE MODELOS ---
rf_model = joblib.load('modelo_hog_random_forest.joblib')
svm_model = joblib.load('svm_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# Directorio para obtener los nombres de las clases
dataset_path = 'recortes_dataset'
class_names = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

def procesar_para_ensamble(img_path):
    # Cargar en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    
    # --- Extracción HOG (para Random Forest) ---
    img_hog = cv2.resize(img, (224, 224))
    hog_detector = cv2.HOGDescriptor(_winSize=(224,224), _blockSize=(32,32), 
                                    _blockStride=(16,16), _cellSize=(16,16), _nbins=9)
    feat_hog = hog_detector.compute(img_hog).flatten().reshape(1, -1)
    
    # --- Extracción SIFT + BoW (para SVM) ---
    img_sift = cv2.resize(img, (200, 200))
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_sift, None)
    
    if des is not None:
        n_esperado = kmeans_model.cluster_centers_.shape[1]
        if des.shape[1] != n_esperado:
            des = des[:, :n_esperado]
        
        des_final = np.ascontiguousarray(des, dtype=kmeans_model.cluster_centers_.dtype)
        words = kmeans_model.predict(des_final)
        hist, _ = np.histogram(words, bins=150, range=(0, 150))
        feat_sift = (hist.astype(np.float32) / (np.sum(hist) + 1e-7)).reshape(1, -1)
    else:
        feat_sift = np.zeros((1, 150), dtype=np.float32)
    
    # Retornamos la imagen original (BGR para mostrar) y los vectores
    img_color = cv2.imread(img_path)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return img_color, (feat_hog, feat_sift)

def confianza_modelo(modelo, X):
    proba = modelo.predict_proba(X)[0]
    idx = np.argmax(proba)
    # Algunos modelos regresan el índice, otros la etiqueta. 
    # Aseguramos retornar el índice de la clase y la probabilidad.
    return idx, float(proba[idx])

def predecir_y_mostrar(img_path):
    img_visualizar, features = procesar_para_ensamble(img_path)
    
    if features is None:
        print("Error: No se pudo cargar la imagen.")
        return

    f_hog, f_sift = features

    # Obtener predicciones y confianzas
    idx_rf, conf_rf = confianza_modelo(rf_model, f_hog)
    idx_svm, conf_svm = confianza_modelo(svm_model, f_sift)

    # Lógica de Ensamble
    if idx_rf == idx_svm:
        clase_final = class_names[idx_rf]
        metodo = "Acuerdo Unánime"
        conf_final = conf_rf
    else:
        if conf_rf >= conf_svm:
            clase_final = class_names[idx_rf]
            metodo = f"Ganador: Random Forest ({round(conf_rf*100, 2)}%)"
        else:
            clase_final = class_names[idx_svm]
            metodo = f"Ganador: SVM ({round(conf_svm*100, 2)}%)"

    # --- Visualización ---
    plt.figure(figsize=(8, 6))
    plt.imshow(img_visualizar)
    plt.title(f"Predicción: {clase_final}\n {metodo}")
    plt.axis('off')
    
    # Texto informativo en la parte inferior
    info_text = (f"RF Pred: {class_names[idx_rf]} ({round(conf_rf,2)})\n"
                 f"SVM Pred: {class_names[idx_svm]} ({round(conf_svm,2)})")
    plt.figtext(0.5, 0.01, info_text, wrap=True, horizontalalignment='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    
    plt.show()

    print(f"--- Resultado para: {os.path.basename(img_path)} ---")
    print(f"Predicción final: {clase_final}")
    print(f"Confianza RF: {conf_rf:.4f} | Confianza SVM: {conf_svm:.4f}")

# --- PRUEBA INDIVIDUAL ---
# Reemplaza con la ruta de la imagen que quieras probar
ruta_prueba = "rebanada.png"
predecir_y_mostrar(ruta_prueba)