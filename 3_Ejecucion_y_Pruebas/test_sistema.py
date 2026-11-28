import io
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


def test_landing(cliente):
    r = cliente.get('/')
    assert r.status_code == 200
    assert 'href="/sistema"' in r.get_data(as_text=True)


def test_pagina_inicial(cliente):
    r = cliente.get('/sistema')
    assert r.status_code == 200
    assert 'Analiza el potencial de un estudiante' in r.get_data(as_text=True)


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


def subir(cliente, contenido, nombre):
    return cliente.post('/cargar', data={'archivo': (io.BytesIO(contenido), nombre)},
                        content_type='multipart/form-data')


def test_descarga_de_plantilla(cliente):
    r = cliente.get('/plantilla.xlsx')
    assert r.status_code == 200
    assert r.data[:2] == b'PK'


def test_carga_de_la_plantilla_en_excel(cliente):
    plantilla = cliente.get('/plantilla.xlsx').data
    r = subir(cliente, plantilla, 'plantilla.xlsx')
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'Estudiante 2' in html
    assert 'Con errores: <b>0</b>' in html


def test_carga_de_csv_con_columnas_en_espanol(cliente):
    csv = ('Nombre;Promedio;Programación;Comunicación;Trabajo en equipo;Prácticas;Exposición\n'
           'Ana;7,5;5;5;5;2;5\n').encode('utf-8')
    r = subir(cliente, csv, 'notas.csv')
    assert r.status_code == 200
    assert 'Ana' in r.get_data(as_text=True)


def test_carga_marca_las_filas_con_datos_invalidos(cliente):
    csv = ('Estudiante,GPA,Programming_Skill,Communication_Skill,Teamwork,Internship_Count,Industry_Exposure\n'
           'Ana,7.5,5,5,5,2,5\nLuis,15,5,5,5,2,5\n').encode('utf-8')
    html = subir(cliente, csv, 'notas.csv').get_data(as_text=True)
    assert 'Con errores: <b>1</b>' in html


def test_carga_rechaza_archivo_sin_columnas_requeridas(cliente):
    r = subir(cliente, b'a,b\n1,2\n', 'malo.csv')
    assert r.status_code == 400
    assert 'Faltan columnas' in r.get_data(as_text=True)


def test_carga_rechaza_formato_no_admitido(cliente):
    r = subir(cliente, b'texto', 'notas.txt')
    assert r.status_code == 400
    assert 'Formato no admitido' in r.get_data(as_text=True)


def test_carga_sin_archivo(cliente):
    r = cliente.post('/cargar', data={}, content_type='multipart/form-data')
    assert r.status_code == 400


def test_archivo_de_un_alumno_muestra_resultado_completo(cliente):
    csv = 'Estudiante,GPA,Programming_Skill,Communication_Skill,Teamwork,Internship_Count,Industry_Exposure\nAna,9.2,8,8,8,3,8\n'
    html = subir(cliente, csv.encode(), 'alumno.csv').get_data(as_text=True)
    assert 'Competitividad Alta' in html
    assert 'Recomendaciones para mejorar' in html
    assert 'Ana' in html


def test_subir_archivo_desde_evaluar_estudiante(cliente):
    csv = 'GPA,Programming_Skill,Communication_Skill,Teamwork,Internship_Count,Industry_Exposure\n15,8,8,8,3,8\n'
    r = cliente.post('/cargar', data={'archivo': (io.BytesIO(csv.encode()), 'alumno.csv'), 'desde': 'evaluar'},
                     content_type='multipart/form-data')
    assert r.status_code == 400
    assert 'el valor debe estar entre' in r.get_data(as_text=True)
