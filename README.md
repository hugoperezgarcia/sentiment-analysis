# Análisis de Sentimiento de Reseñas de Películas
Este proyecto se centra en el Análisis de Sentimiento de un conjunto de datos de reseñas de películas, comparando dos enfoques distintos: un modelo de machine learning tradicional (Regresión Logística con TF-IDF) y un modelo basado en transformers preentrenado (BERT en español).

## Estructura y Metodología
### Preparación de Datos (EDA)
Carga y Filtrado: Se carga el archivo film_reviews_result.csv y se filtra para conservar únicamente las columnas de review_rate (valoración) y review_text (texto de la reseña).

Clasificación de Sentimiento: Se crea una etiqueta binaria label_txt ("positivo" o "negativo") a partir de la valoración (review_rate), considerando:

Valoracion ≤ 5 → Negativo

Valoracion >5 → Positivo

Muestreo: Para la fase de modelado inicial, se utiliza una muestra de 1,000 reseñas del dataset original, aunque se conserva la versión completa para usos posteriores.

Análisis Descriptivo: Se confirma un desbalanceo entre las clases positivo/negativo en el subconjunto de datos.

## Modelos de Análisis de Sentimiento
### Modelo Sencillo (Regresión Logística con BoW/TF-IDF)
Técnica: Se implementa un pipeline que combina la extracción de características mediante TfidfVectorizer y la predicción con LogisticRegression.

Preprocesamiento: Se utiliza un tokenizer personalizado basado en spaCy (es_core_news_sm) para realizar lematización y eliminar stop words en español, mejorando la representación del texto.

Estrategia: Se aplica class_weight="balanced" en la Regresión Logística para mitigar el efecto del desbalanceo de clases.

Limitación Identificada: El modelo tiene dificultades con la negación y el contexto de la frase (ej. "No es nada mala"), ya que se enfoca solo en palabras individuales.

### Modelo Complejo (BERT Preentrenado)
Técnica: Se utiliza un modelo BERT preentrenado en español (verotei/bert-base-spanish-wwm-cased-nlp-transformers-sentiment-amazon) de la librería Hugging Face transformers. Este modelo aprovecha el contexto de la frase.

Implementación: Se definen funciones para la tokenización y predicción, que devuelven las probabilidades de sentimiento positivo y negativo.

Evaluación: Se evalúa el modelo preentrenado directamente en el set de prueba sin aplicar fine-tuning para una comparación inicial de rendimiento.

Resultados: Aunque los resultados de las métricas automáticas fueron sorprendentemente bajos, las pruebas manuales sugieren una notable superioridad en la comprensión del contexto y la ironía (salvo en casos muy extremos).

## Resultados y conclusiones
Los modelos con transformers muestran un potencial superior para capturar matices lingüísticos como la negación y el sarcasmo, elementos clave en el análisis de reseñas. El siguiente paso recomendado sería aplicar fine-tuning al modelo BERT con el dataset específico de reseñas para maximizar su precisión.
