import io
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_file

import sistema

app = Flask(__name__)
MODELO = sistema.cargar()
DATOS = sistema.resumen(MODELO)


def contexto(**extra):
    base = {'variables': sistema.VARIABLES, 'etiquetas': sistema.ETIQUETAS,
            'limites': sistema.LIMITES, 'valores': {}, 'errores': [], 'resultado': None,
            'lote': None, 'error_lote': None, 'nombre_archivo': '', 'alumno': '', 'origen': 'Individual', 'desde': '',
            'metricas': MODELO['metricas'], 'datos': DATOS,
            'info': {'epoca': MODELO['epoca'], 'registros': MODELO['registros']}}
    base.update(extra)
    if base['resultado'] or base['lote']:
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
    return render_template('index.html', **contexto(valores=valores, resultado=sistema.predecir(MODELO, valores),
                                                    alumno=alumno, origen='Archivo' if alumno else 'Individual'))


@app.post('/api/predecir')
def predecir_api():
    datos = request.get_json(silent=True) or {}
    valores, errores = sistema.validar(datos)
    if errores:
        return jsonify({'errores': errores}), 400
    return jsonify(sistema.predecir(MODELO, valores))


@app.post('/cargar')
def cargar_archivo():
    archivo = request.files.get('archivo')
    desde = request.form.get('desde', '')
    if archivo is None or not archivo.filename:
        return render_template('index.html', **contexto(error_lote='Seleccione un archivo .xlsx o .csv.', desde=desde)), 400
    try:
        tabla, columnas = sistema.leer_tabla(archivo, archivo.filename)
    except Exception as e:
        return render_template('index.html', **contexto(error_lote=str(e), desde=desde)), 400
    lote = sistema.evaluar_tabla(MODELO, tabla, columnas)
    if desde == 'evaluar' and len(lote) == 1 and lote[0]['errores']:
        return render_template('index.html', **contexto(error_lote=' '.join(lote[0]['errores']), desde=desde)), 400
    if len(lote) == 1 and not lote[0]['errores']:
        return render_template('index.html', **contexto(valores=lote[0]['valores'], alumno=lote[0]['estudiante'],
                                                        resultado=sistema.predecir(MODELO, lote[0]['valores']),
                                                        origen='Archivo', nombre_archivo=archivo.filename))
    return render_template('index.html', **contexto(lote=lote, nombre_archivo=archivo.filename))


@app.get('/plantilla.xlsx')
def descargar_plantilla():
    salida = io.BytesIO()
    tabla = sistema.plantilla()
    if request.args.get('alumnos') == '1':
        tabla = tabla.head(1).replace({'Estudiante 1': 'Nombre del alumno'})
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
