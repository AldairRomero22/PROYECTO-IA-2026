import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2_Codigo'))

import app as aplicacion
import sistema

PERFIL_ALTO = {'GPA': 9.4, 'Programming_Skill': 9, 'Communication_Skill': 8,
               'Teamwork': 8, 'Internship_Count': 3, 'Industry_Exposure': 9}
PERFIL_BAJO = {'GPA': 6.3, 'Programming_Skill': 2, 'Communication_Skill': 3,
               'Teamwork': 3, 'Internship_Count': 0, 'Industry_Exposure': 1}


@pytest.fixture(scope='module')
def cliente():
    aplicacion.app.config['TESTING'] = True
    return aplicacion.app.test_client()


def test_pagina_inicial(cliente):
    r = cliente.get('/')
    assert r.status_code == 200
    assert 'Perfil del estudiante' in r.get_data(as_text=True)


def test_salud(cliente):
    r = cliente.get('/salud')
    assert r.status_code == 200
    assert r.get_json()['estado'] == 'activo'


def test_prediccion_perfil_alto(cliente):
    r = cliente.post('/api/predecir', json=PERFIL_ALTO)
    assert r.status_code == 200
    assert r.get_json()['clase'] == 'High'


def test_prediccion_perfil_bajo(cliente):
    r = cliente.post('/api/predecir', json=PERFIL_BAJO)
    assert r.status_code == 200
    assert r.get_json()['clase'] == 'Low'


def test_probabilidades_suman_uno(cliente):
    prob = cliente.post('/api/predecir', json=PERFIL_ALTO).get_json()['probabilidades']
    assert abs(sum(prob.values()) - 1) < 1e-3


def test_recomendaciones_perfil_bajo(cliente):
    rec = cliente.post('/api/predecir', json=PERFIL_BAJO).get_json()['recomendaciones']
    assert len(rec) == 3
    assert all(r['actual'] < r['objetivo'] for r in rec)


def test_sin_recomendaciones_si_supera_clase_alta(cliente):
    perfil = {'GPA': 10, 'Programming_Skill': 10, 'Communication_Skill': 10,
              'Teamwork': 10, 'Internship_Count': 4, 'Industry_Exposure': 10}
    assert cliente.post('/api/predecir', json=perfil).get_json()['recomendaciones'] == []


@pytest.mark.parametrize('campo,valor', [('GPA', 11), ('GPA', -1), ('Programming_Skill', 0),
                                         ('Teamwork', 'abc'), ('Internship_Count', 2.5),
                                         ('Industry_Exposure', None)])
def test_rechaza_datos_invalidos(cliente, campo, valor):
    datos = dict(PERFIL_ALTO)
    datos[campo] = valor
    r = cliente.post('/api/predecir', json=datos)
    assert r.status_code == 400
    assert r.get_json()['errores']


def test_rechaza_datos_faltantes(cliente):
    r = cliente.post('/api/predecir', json={'GPA': 8})
    assert r.status_code == 400
    assert len(r.get_json()['errores']) == 5


def test_formulario_web(cliente):
    r = cliente.post('/predecir', data=PERFIL_ALTO)
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'Competitividad Alta' in html
    assert 'Recomendaciones para mejorar' in html


def test_formulario_con_error_muestra_mensaje(cliente):
    r = cliente.post('/predecir', data={**PERFIL_ALTO, 'GPA': '15'})
    assert r.status_code == 400
    assert 'Revise los datos ingresados' in r.get_data(as_text=True)


def test_rnf02_tiempo_de_respuesta(cliente):
    tiempos = []
    for _ in range(30):
        t0 = time.perf_counter()
        cliente.post('/api/predecir', json=PERFIL_ALTO)
        tiempos.append(time.perf_counter() - t0)
    assert max(tiempos) < 5


def test_rnf03_f1_macro_minimo():
    metricas = aplicacion.MODELO['metricas']['Red neuronal (MLP)']
    assert metricas['f1_macro'] >= 0.70


def test_guardado_y_carga_del_sistema(tmp_path):
    ruta = tmp_path / 'copia.joblib'
    sistema.guardar(aplicacion.MODELO, ruta)
    copia = sistema.cargar(ruta)
    a = sistema.predecir(aplicacion.MODELO, PERFIL_ALTO)['probabilidades']
    b = sistema.predecir(copia, PERFIL_ALTO)['probabilidades']
    assert a == b


def test_recuperacion_si_el_archivo_esta_danado(tmp_path):
    ruta = tmp_path / 'danado.joblib'
    ruta.write_bytes(b'no es un modelo')
    assert sistema.cargar(ruta)['registros'] == 2000
