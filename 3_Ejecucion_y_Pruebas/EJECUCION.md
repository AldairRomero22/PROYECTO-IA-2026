# Ejecución y pruebas del sistema

## Requisitos
Python 3.10 o superior y las librerías de `requirements.txt` (scikit-learn 1.8.0).

## Pasos
1. `pip install -r requirements.txt`
2. `python 2_Codigo/entrenar.py` entrena los modelos y guarda el sistema en `2_Codigo/modelo/sistema.joblib`. Si el archivo no existe o fue creado con otra versión de scikit-learn, la aplicación lo vuelve a entrenar sola al iniciar (unos 4 segundos).
3. `python 2_Codigo/app.py` inicia la aplicación web en el puerto 5000.
4. `python -m pytest 3_Ejecucion_y_Pruebas -v` ejecuta las 20 pruebas automáticas.

## Uso de la API
```bash
curl -X POST http://localhost:5000/api/predecir -H "Content-Type: application/json" \
  -d '{"GPA":7.2,"Programming_Skill":4,"Communication_Skill":5,"Teamwork":6,"Internship_Count":1,"Industry_Exposure":3}'
```
Devuelve la clase estimada, las probabilidades, la segunda opinión, las recomendaciones y la comparación con cada nivel. Los datos inválidos devuelven el código 400 con la lista de errores.

## Despliegue en la nube (Render)
1. Subir el repositorio a GitHub.
2. En Render crear un *Web Service* desde el repositorio.
3. Comando de compilación: `pip install -r requirements.txt`. Comando de inicio: `gunicorn --chdir 2_Codigo app:app`.

## Pruebas incluidas
Página inicial, estado del servicio, predicción de perfiles alto y bajo, probabilidades que suman 1, recomendaciones, rechazo de datos inválidos o faltantes, formulario web, tiempo de respuesta menor a 5 s (RNF-02), F1-macro mínimo de 0.70 (RNF-03), guardado y carga del sistema y recuperación ante un archivo dañado.
