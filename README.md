# Emplea360 · Sistema inteligente para la evaluación y predicción de la competitividad laboral de estudiantes universitarios

Curso ISIA-108 Inteligencia Artificial · Universidad Privada Antenor Orrego (UPAO), Trujillo.

**Integrantes:** Aguirre Anampa Jayro Paul · Romero Asmat Bryans Aldair · Rujel Álvarez Kevin David · Salvatierra Prieto Gabriel del Piero

El sistema recibe el perfil de un estudiante (GPA, programación, comunicación, trabajo en equipo, prácticas y exposición laboral) y estima su nivel de competitividad laboral (baja, media o alta) con una red neuronal, usando una regresión logística como segunda opinión. Además genera recomendaciones de mejora, compara el perfil con el promedio de cada nivel y permite cargar un Excel o CSV con varios estudiantes para evaluarlos de una vez.

## Contenido del repositorio

| Carpeta | Contenido |
|---|---|
| `1_Documentacion` | Informe del proyecto y notebook completo de Google Colab |
| `2_Codigo` | Código del sistema, aplicación web y conjunto de datos |
| `3_Ejecucion_y_Pruebas` | Pruebas automáticas, capturas y guía de ejecución |

## Ejecución rápida

```bash
pip install -r requirements.txt
python 2_Codigo/entrenar.py
python 2_Codigo/app.py
```

La aplicación queda disponible en http://localhost:5000. Las pruebas se ejecutan con `python -m pytest 3_Ejecucion_y_Pruebas`.

## Resultados (conjunto de prueba, 300 estudiantes)

| Modelo | Exactitud | F1-macro |
|---|---|---|
| Red neuronal (MLP) | 0.9667 | 0.9012 |
| Regresión logística | 0.9467 | 0.8507 |

La etiqueta del conjunto de datos se calcula con una fórmula a partir de las variables, por lo que estas métricas validan el método de trabajo y no garantizan el mismo desempeño con datos reales.
