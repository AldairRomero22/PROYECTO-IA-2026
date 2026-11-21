import os

from flask import Flask, jsonify, render_template, request

import sistema

app = Flask(__name__)
MODELO = sistema.cargar()


def contexto(**extra):
    base = {'variables': sistema.VARIABLES, 'etiquetas': sistema.ETIQUETAS,
            'limites': sistema.LIMITES, 'valores': {}, 'errores': [], 'resultado': None,
            'metricas': MODELO['metricas']}
    base.update(extra)
    return base


@app.get('/')
def inicio():
    return render_template('index.html', **contexto())


@app.post('/predecir')
def predecir_formulario():
    valores, errores = sistema.validar(request.form)
    if errores:
        return render_template('index.html', **contexto(valores=request.form, errores=errores)), 400
    resultado = sistema.predecir(MODELO, valores)
    return render_template('index.html', **contexto(valores=valores, resultado=resultado))


@app.post('/api/predecir')
def predecir_api():
    datos = request.get_json(silent=True) or {}
    valores, errores = sistema.validar(datos)
    if errores:
        return jsonify({'errores': errores}), 400
    return jsonify(sistema.predecir(MODELO, valores))


@app.get('/salud')
def salud():
    return jsonify({'estado': 'activo', 'registros_entrenamiento': MODELO['registros'],
                    'metricas': MODELO['metricas']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
