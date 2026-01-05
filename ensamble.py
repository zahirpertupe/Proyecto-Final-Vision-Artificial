import cv2
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#cargamos el RF y el svm
rf_model = joblib.load('modelo_hog_random_forest.joblib')
svm_model = joblib.load('svm_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
dataset_path = 'recortes_dataset'
# Ambos modelos fueron entreados así
class_names = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])
class_to_id = {name: i for i, name in enumerate(class_names)}


def procesar_para_ensamble(img_path):
    # Pasamos a escala de grises pq el dataset está en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Usamos HOG co los mismos parametros que usamos en Rmdom foreste
    img_hog = cv2.resize(img, (224, 224))
    hog_detector = cv2.HOGDescriptor(_winSize=(224,224), _blockSize=(32,32), 
                                    _blockStride=(16,16), _cellSize=(16,16), _nbins=9)
    feat_hog = hog_detector.compute(img_hog).flatten().reshape(1, -1)
    
    #de igual forma, svm con los mismos parametros
    img_sift = cv2.resize(img, (200, 200))
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_sift, None)
    
    if des is not None:
        n_esperado = kmeans_model.cluster_centers_.shape[1]
    
        if des.shape[1] != n_esperado:
            des = des[:, :n_esperado]
    

        des_final = np.ascontiguousarray(des,dtype=kmeans_model.cluster_centers_.dtype)
    
        words = kmeans_model.predict(des_final)
    
        hist, _ = np.histogram(words, bins=150, range=(0, 150))
        feat_sift = (hist.astype(np.float32) / (np.sum(hist) + 1e-7)).reshape(1, -1)
    else:
        feat_sift = np.zeros((1, 150), dtype=np.float32)
    
    #regresamos los "vectores de caracteristica" de ambos modelos
    return feat_hog, feat_sift

#entrenamos a svm con probability = true
def confianza_modelo(modelo, X):
    proba = modelo.predict_proba(X)[0]
    idx = np.argmax(proba)
    return modelo.classes_[idx], float(proba[idx])


#Hacaemos la predicción
def predecir_pan(img_path):
    features = procesar_para_ensamble(img_path)
    if features is None:
        return "Imagen no encontrada"

    f_hog, f_sift = features

    # Predicciones 
    pred_rf = rf_model.predict(f_hog)[0]
    pred_svm = svm_model.predict(f_sift)[0]

    # Si ambos están de acuero
    if pred_rf == pred_svm:
        return {
            "Random Forest": pred_rf,
            "SVM": pred_svm,
            "Resultado Final": pred_rf,
            "Confianza": "Alta (coinciden)"
        }

    #Si no, elegimos al de mayor confianza 
    _, conf_rf = confianza_modelo(rf_model, f_hog)
    _, conf_svm = confianza_modelo(svm_model, f_sift)

    if conf_rf >= conf_svm:
        origen = "Random Forest"
        verdadero = pred_rf
    else:
        origen = "SVM"
        verdadero = pred_svm
 

    return {
    "Random Forest": rf_model.classes_[pred_rf],
    "SVM": svm_model.classes_[pred_svm],
   "Resultado Final": verdadero,
    "Modelo ganador": origen,
    "Confianza RF": round(conf_rf, 4),
    "Confianza SVM": round(conf_svm, 4)
} 

# Evaluamos el ensamble con los mismos archivos
def evaluar_ensamble(test_dir):
    y_true = []
    y_pred = []


    for clase in class_names:
        clase_path = os.path.join(test_dir, clase)

        for img_name in os.listdir(clase_path):
            img_path = os.path.join(clase_path, img_name)

            resultado = predecir_pan(img_path)

            if isinstance(resultado, str):
                continue

            y_true.append(class_to_id[clase])   # número
            y_pred.append(resultado["Resultado Final"])  # número

    return y_true, y_pred


test_dir = "valid"

y_true, y_pred = evaluar_ensamble(test_dir)

acc = accuracy_score(y_true, y_pred)
print("Accuracy Ensamble:", acc)

labels = list(range(len(class_names)))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)


plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión del Ensamble")
plt.show()



print("Accuracy Ensamble:", accuracy_score(y_true, y_pred))

print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

#predicción individual
print(predecir_pan("concha.png"))