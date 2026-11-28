# Ejecución y pruebas del sistema

## Requisitos
Python 3.10 o superior y las librerías de `requirements.txt` (scikit-learn 1.8.0).

## Pasos
1. `pip install -r requirements.txt`
2. `python 2_Codigo/entrenar.py` entrena los modelos y guarda el sistema en `2_Codigo/modelo/sistema.joblib`. Si el archivo no existe o fue creado con otra versión de scikit-learn, la aplicación lo vuelve a entrenar sola al iniciar (unos 4 segundos).
3. `python 2_Codigo/app.py` inicia la aplicación web en el puerto 5000.
4. `python -m pytest 3_Ejecucion_y_Pruebas -v` ejecuta las 29 pruebas automáticas.

## Carga de archivos
La dirección principal muestra una página de presentación con un botón que lleva al sistema, que vive en /sistema. La interfaz tiene seis módulos en la barra lateral: Panel de control, Evaluar estudiante, Resultados, Historial, Conjunto de datos y Acerca del sistema. El historial se guarda en el navegador del usuario. El botón con forma de gota de la barra superior cambia entre la vista clásica y la vista Liquid Glass, con paneles de vidrio esmerilado sobre un fondo oscuro de ondas metálicas y animaciones. La página de presentación usa GSAP y ScrollTrigger y Lenis para el scroll suave (cargados desde CDN) para animar en 3D las secciones de la página de presentación al hacer scroll; la elección también se recuerda en el navegador.

En el módulo *Evaluar estudiante* se llena el formulario o se sube el archivo de un solo alumno, un Excel (.xlsx) o un CSV con las columnas GPA, Programming_Skill, Communication_Skill, Teamwork, Internship_Count e Industry_Exposure (también se aceptan sus nombres en español) y, opcionalmente, una columna Estudiante. La plantilla se descarga desde el mismo módulo y los datos no se guardan. El módulo *Resultados* muestra el nivel, el gráfico comparativo, las recomendaciones y cuatro tarjetas de empresas reales según la carrera, junto al gráfico, cada una con su logo y un botón que lleva a su portal de empleo para enviar el CV. La carrera se elige en el formulario o va en la columna opcional Carrera del archivo; si falta, se estima por las habilidades más fuertes. El logo se carga desde internet en el navegador del usuario.

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

El módulo *Resultados* incluye además un Coach de carrera con IA que genera un plan de 4 semanas con Claude. Solo aparece si el servidor tiene la variable de entorno ANTHROPIC_API_KEY (en Render, en Environment; en local, por ejemplo `set ANTHROPIC_API_KEY=...` antes de `python app.py`). Opcionalmente se puede cambiar el modelo con ANTHROPIC_MODEL (por defecto claude-haiku-5-5) y el tope diario de planes con COACH_LIMITE_DIARIO (por defecto 150). Cada visitante puede generar hasta 5 planes por hora. Sin la clave, el sistema funciona igual y el coach no se muestra.
