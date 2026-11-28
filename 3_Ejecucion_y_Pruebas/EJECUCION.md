# Ejecución y pruebas del sistema

## Requisitos
Python 3.10 o superior y las librerías de `requirements.txt` (scikit-learn 1.8.0).

## Pasos
1. `pip install -r requirements.txt`
2. `python 2_Codigo/entrenar.py` entrena los modelos y guarda el sistema en `2_Codigo/modelo/sistema.joblib`. Si el archivo no existe o fue creado con otra versión de scikit-learn, la aplicación lo vuelve a entrenar sola al iniciar (unos 4 segundos).
3. `python 2_Codigo/app.py` inicia la aplicación web en el puerto 5000.
4. `python -m pytest 3_Ejecucion_y_Pruebas -v` ejecuta las 29 pruebas automáticas.

## Carga de archivos
La dirección principal muestra una página de presentación con un botón que lleva al sistema, que vive en /sistema. La interfaz tiene siete módulos en la barra lateral: Panel de control, Evaluar estudiante, Evaluación masiva, Resultados, Historial, Conjunto de datos y Acerca del sistema. El historial se guarda en el navegador del usuario. El botón con forma de gota de la barra superior cambia entre la vista clásica y la vista Liquid Glass, con paneles de vidrio esmerilado sobre un fondo oscuro de ondas metálicas y animaciones. La página de presentación usa GSAP y ScrollTrigger y Lenis para el scroll suave (cargados desde CDN) para que las capturas del sistema se expandan al hacer scroll; la elección también se recuerda en el navegador.

En el módulo *Evaluación masiva* se sube un Excel (.xlsx) o un CSV con las columnas GPA, Programming_Skill, Communication_Skill, Teamwork, Internship_Count e Industry_Exposure (también se aceptan sus nombres en español) y, opcionalmente, una columna Estudiante. Las plantillas de un alumno y de varios alumnos se descargan desde el mismo módulo. Si el archivo tiene un solo alumno, se muestra directamente su resultado completo; con varios, cada fila tiene el botón «Ver detalle». El módulo *Evaluar estudiante* también permite subir el archivo de un alumno además de llenar el formulario. Cada fila recibe su nivel, la probabilidad de la clase Alta y la variable prioritaria de mejora; los resultados se pueden descargar en CSV. El máximo es de 500 estudiantes por archivo y los datos no se guardan.

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
Página inicial, estado del servicio, descarga de plantilla, carga de Excel y CSV (archivo de un solo alumno, columnas en inglés y español, filas con datos inválidos, columnas faltantes, formato no admitido), predicción de perfiles alto y bajo, probabilidades que suman 1, recomendaciones, rechazo de datos inválidos o faltantes, formulario web, tiempo de respuesta menor a 5 s (RNF-02), F1-macro mínimo de 0.70 (RNF-03), guardado y carga del sistema y recuperación ante un archivo dañado.
