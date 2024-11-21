import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Cargar el DataFrame con datos históricos de vendedores exitosos y no exitosos
df_hist = pd.read_excel('vendedores_historico.xlsx')

# Preprocesar datos
label_encoder = LabelEncoder()
df_hist['educacion'] = label_encoder.fit_transform(df_hist['educacion'])

# Definir variables
X = df_hist.drop(columns=['nombre', 'exito'])
y = df_hist['exito']

# Separar datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo con regularización adicional
model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de entrenamiento
y_pred_train = model.predict(X_train)
report_train = classification_report(y_train, y_pred_train, output_dict=True)
matrix_train = confusion_matrix(y_train, y_pred_train)

# Evaluar el modelo en el conjunto de prueba
y_pred_test = model.predict(X_test)
report_test = classification_report(y_test, y_pred_test, output_dict=True)
matrix_test = confusion_matrix(y_test, y_pred_test)

# Imprimir la evaluación del modelo en la consola
print("Evaluación del Modelo en Entrenamiento:")
print(f"Precisión: {report_train['accuracy']*100:.2f}%")
print("Reporte de Clasificación:")
print(f"Precisión Positivo: {report_train['1']['precision']:.2f}")
print(f"Recall Positivo: {report_train['1']['recall']:.2f}")
print(f"F1-score Positivo: {report_train['1']['f1-score']:.2f}")
print(f"Precisión Negativo: {report_train['0']['precision']:.2f}")
print(f"Recall Negativo: {report_train['0']['recall']:.2f}")
print(f"F1-score Negativo: {report_train['0']['f1-score']:.2f}")
print(f"\nMatriz de Confusión:\n{matrix_train}\n")

print("Evaluación del Modelo en Prueba:")
print(f"Precisión: {report_test['accuracy']*100:.2f}%")
print("Reporte de Clasificación:")
print(f"Precisión Positivo: {report_test['1']['precision']:.2f}")
print(f"Recall Positivo: {report_test['1']['recall']:.2f}")
print(f"F1-score Positivo: {report_test['1']['f1-score']:.2f}")
print(f"Precisión Negativo: {report_test['0']['precision']:.2f}")
print(f"Recall Negativo: {report_test['0']['recall']:.2f}")
print(f"F1-score Negativo: {report_test['0']['f1-score']:.2f}")
print(f"\nMatriz de Confusión:\n{matrix_test}\n")

# Validación Cruzada
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Precisión de Validación Cruzada: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

# Función para procesar currículum y realizar predicciones
def process_cv(file_path):
    cv = pd.read_excel(file_path)
    cv['educacion'] = label_encoder.transform(cv['educacion'])
    X_cv = cv.drop(columns=['nombre'])
    predictions = model.predict(X_cv)
    cv['resultado'] = ['Aprobado' if pred == 1 else 'No Aprobado' for pred in predictions]
    return cv

# Función para cargar currículum, realizar predicciones y guardar resultados en Excel
def load_cv():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if file_path:
        df_result = process_cv(file_path)
        result_file_path = 'resultados_evaluacion.xlsx'
        df_result.to_excel(result_file_path, index=False)
        messagebox.showinfo("Resultados de Evaluación", f"Resultados guardados en {result_file_path}")

# Crear la interfaz gráfica
def create_gui():
    root = tk.Tk()
    root.title("Evaluación de Currículos")
    root.geometry("400x250")

    label = tk.Label(root, text="Evaluación de Currículos para Vendedor", font=("Helvetica", 16))
    label.pack(pady=10)

    cv_button = ttk.Button(root, text="Cargar CVs y Evaluar", command=load_cv)
    cv_button.pack(pady=10)

    root.mainloop()

# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    create_gui()
