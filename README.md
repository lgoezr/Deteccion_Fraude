# Deteccion_Fraude

Este es un sistema de detección de fraude en órdenes de e-commerce con Machine Learning. El proyecto limpia y preprocesa un dataset de 25,000 transacciones desbalanceadas (3.06% de fraude), selecciona variables con evidencia estadística (VIF, Chi-cuadrado) y entrena modelos de Regresión Logística y Random Forest para clasificar el riesgo de cada orden, evaluados con métricas robustas para clases desbalanceadas (Precision, Recall, F1, ROC-AUC, PR-AUC).
 
## Dataset
 
- **Fuente:** [E-commerce Customer Behavior and Logistics Dataset](https://www.kaggle.com/datasets/deeplumiere/e-commerce-customer-behavior-and-logistics-dataset) (Kaggle)
- **Tamaño:** 25,000 órdenes, 26 variables
- **Variable objetivo:** `fraud_flag` (Yes/No)
- **Desbalance de clases:** 3.06% de fraude (766 de 25,000 órdenes)
## Metodología
 
1. **EDA manual y automático:** estadísticos descriptivos (media, mediana, IQR, outliers), visualizaciones (boxplots, histogramas, mapas de calor de correlación) y perfilamiento automatizado con `ydata-profiling`, comparando explícitamente ambos enfoques.
2. **Detección de calidad de datos:** identificación de inconsistencias (edades imposibles, pesos negativos, distancias de envío irreales, un `order_id` duplicado, una columna de valor total que no coincidía con su fórmula esperada).
3. **Partición estratificada 60/20/20** (train/val/test) antes de cualquier limpieza, para evitar fuga de datos, con exclusión explícita de columnas posteriores al evento (leakage temporal).
4. **Comparación experimental controlada:** un dataset "sucio" (limpieza mínima) frente a un dataset "limpio" (outliers tratados, inconsistencias corregidas, codificación justificada por tipo de variable, selección de variables con VIF y prueba Chi-cuadrado, y escalamiento log1p + RobustScaler).
5. **Modelado:** Regresión Logística (`class_weight="balanced"`) sobre ambos escenarios, más un Random Forest sobre el dataset limpio.
6. **Evaluación:** métricas apropiadas para clases desbalanceadas — Precision, Recall, F1, ROC-AUC y PR-AUC — en lugar de depender solo de accuracy.
 
## Tecnologías utilizadas
 
- Python
- pandas, numpy
- scikit-learn
- statsmodels (VIF)
- scipy (prueba Chi-cuadrado)
- matplotlib, seaborn
- ydata-profiling

