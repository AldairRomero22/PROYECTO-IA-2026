import io
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_file

import coach
import sistema

app = Flask(__name__)
MODELO = sistema.cargar()
DATOS = sistema.resumen(MODELO)


def contexto(**extra):
    base = {'variables': sistema.VARIABLES, 'etiquetas': sistema.ETIQUETAS,
            'limites': sistema.LIMITES, 'valores': {}, 'errores': [], 'resultado': None,
            'error_lote': None, 'nombre_archivo': '', 'alumno': '', 'origen': 'Individual',
            'metricas': MODELO['metricas'], 'datos': DATOS, 'carreras': list(sistema.CARRERAS), 'coach_activo': coach.disponible(),
            'info': {'epoca': MODELO['epoca'], 'registros': MODELO['registros']}}
    base.update(extra)
    if base['resultado']:
        base['marca'] = uuid.uuid4().hex
    return base


@app.get('/')
def inicio():
    return render_template('landing.html', datos=DATOS)


@app.get('/sistema')
def sistema_web():
    return render_template('index.html', **contexto())


@app.post('/predecir')
def predecir_formulario():
    valores, errores = sistema.validar(request.form)
    if errores:
        return render_template('index.html', **contexto(valores=request.form, errores=errores)), 400
    alumno = request.form.get('alumno', '')[:80]
    return render_template('index.html', **contexto(valores=valores, resultado=sistema.predecir(MODELO, valores, request.form.get('carrera', '')),
                                                    alumno=alumno, origen='Archivo' if alumno else 'Individual'))


@app.post('/api/predecir')
def predecir_api():
    datos = request.get_json(silent=True) or {}
    valores, errores = sistema.validar(datos)
    if errores:
        return jsonify({'errores': errores}), 400
    return jsonify(sistema.predecir(MODELO, valores))


def rechazo(mensaje):
    return render_template('index.html', **contexto(error_lote=mensaje)), 400


@app.post('/api/coach')
def coach_api():
    if not coach.disponible():
        return jsonify({'error': 'El coach con IA no está configurado.'}), 503
    datos = request.get_json(silent=True) or {}
    valores, errores = sistema.validar(datos)
    if errores:
        return jsonify({'errores': errores}), 400
    ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
    if not coach.permitir(ip):
        return jsonify({'error': 'Alcanzaste el límite de planes por ahora. Inténtalo más tarde.'}), 429
    resultado = sistema.predecir(MODELO, valores, datos.get('carrera', ''))
    try:
        return jsonify(coach.generar(resultado, resultado['ofertas']['carrera']))
    except Exception:
        return jsonify({'error': 'No se pudo generar el plan. Inténtalo de nuevo.'}), 502


@app.post('/cargar')
def cargar_archivo():
    archivo = request.files.get('archivo')
    if archivo is None or not archivo.filename:
        return rechazo('Seleccione un archivo .xlsx o .csv.')
    try:
        tabla, columnas = sistema.leer_tabla(archivo, archivo.filename)
    except Exception as e:
        return rechazo(str(e))
    lote = sistema.evaluar_tabla(MODELO, tabla, columnas)
    if len(lote) != 1:
        return rechazo('El archivo debe contener los datos de un solo alumno.')
    if lote[0]['errores']:
        return rechazo(' '.join(lote[0]['errores']))
    return render_template('index.html', **contexto(valores={**lote[0]['valores'], 'carrera': lote[0]['carrera']}, alumno=lote[0]['estudiante'],
                                                    resultado=sistema.predecir(MODELO, lote[0]['valores'], lote[0]['carrera']),
                                                    origen='Archivo', nombre_archivo=archivo.filename))


@app.get('/plantilla.xlsx')
def descargar_plantilla():
    salida = io.BytesIO()
    tabla = sistema.plantilla().head(1).replace({'Estudiante 1': 'Nombre del alumno'})
    tabla.to_excel(salida, index=False)
    salida.seek(0)
    return send_file(salida, as_attachment=True, download_name='plantilla_estudiantes.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.get('/salud')
def salud():
    return jsonify({'estado': 'activo', 'registros_entrenamiento': MODELO['registros'],
                    'metricas': MODELO['metricas']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
