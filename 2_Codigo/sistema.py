import copy
import unicodedata

import sklearn
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RAIZ = Path(__file__).resolve().parent
CSV = RAIZ / 'datos' / 'graduate_employment_dataset.csv'
MODELO = RAIZ / 'modelo' / 'sistema.joblib'

VARIABLES = ['GPA', 'Programming_Skill', 'Communication_Skill', 'Teamwork',
             'Internship_Count', 'Industry_Exposure']
CLASES = ['Low', 'Medium', 'High']
NIVELES = {'Low': 'Baja', 'Medium': 'Media', 'High': 'Alta'}
ETIQUETAS = {'GPA': 'Promedio ponderado (GPA)',
             'Programming_Skill': 'Habilidad de programación',
             'Communication_Skill': 'Habilidad de comunicación',
             'Teamwork': 'Trabajo en equipo',
             'Internship_Count': 'Prácticas preprofesionales',
             'Industry_Exposure': 'Exposición al entorno laboral'}
LIMITES = {'GPA': (0.0, 10.0), 'Programming_Skill': (1, 10), 'Communication_Skill': (1, 10),
           'Teamwork': (1, 10), 'Internship_Count': (0, 10), 'Industry_Exposure': (0, 10)}
CORTAS = {'GPA': 'Promedio', 'Programming_Skill': 'Programación', 'Communication_Skill': 'Comunicación',
          'Teamwork': 'Equipo', 'Internship_Count': 'Prácticas', 'Industry_Exposure': 'Exposición'}
CONSEJOS = {'GPA': 'Reforzar el rendimiento académico: tutorías, grupos de estudio y repaso de los cursos con menor nota.',
            'Programming_Skill': 'Practicar programación de forma constante con proyectos propios, retos en línea y cursos de especialización.',
            'Communication_Skill': 'Entrenar la comunicación oral y escrita: exposiciones, talleres de oratoria y redacción de informes.',
            'Teamwork': 'Participar en proyectos grupales, hackatones o voluntariados para fortalecer el trabajo colaborativo.',
            'Internship_Count': 'Buscar prácticas preprofesionales o pasantías, aunque sean de corta duración.',
            'Industry_Exposure': 'Acercarse al entorno laboral: charlas con empresas, ferias de empleo, visitas y mentorías con profesionales.'}

CARRERAS = {
    'Ingeniería de Sistemas e IA': ('Practicante de desarrollo, datos o infraestructura TI',
        [('BCP', 'viabcp.com', 'https://www.viabcp.com/unete-al-equipo-bcp/talento-joven'),
         ('Interbank', 'interbank.pe', 'https://pe.indeed.com/cmp/Interbank'),
         ('Globant', 'globant.com', None), ('NTT DATA', 'nttdata.com', None)]),
    'Administración y Negocios': ('Practicante de gestión comercial u operaciones',
        [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus'), ('Alicorp', 'alicorp.com.pe', None),
         ('Falabella', 'falabella.com.pe', None), ('Supermercados Peruanos', 'spsa.pe', None)]),
    'Contabilidad y Finanzas': ('Practicante contable, financiero o de auditoría',
        [('BBVA', 'bbva.pe', None), ('Scotiabank', 'scotiabank.com.pe', None),
         ('KPMG', 'kpmg.com', None), ('PwC', 'pwc.com', None)]),
    'Marketing y Comunicación': ('Practicante de marketing, marca o comunicaciones',
        [('Belcorp', 'belcorp.biz', None), ('Unilever', 'unilever.com', None),
         ('Ripley', 'ripley.com.pe', None), ('Entel Perú', 'entel.pe', None)]),
    'Ingeniería Industrial': ('Practicante de producción, calidad o logística',
        [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus'),
         ('Cementos Pacasmayo', 'cementospacasmayo.com.pe', 'https://pe.indeed.com/cmp/Cementos-Pacasmayo'),
         ('Alicorp', 'alicorp.com.pe', None), ('Grupo Gloria', 'grupogloria.com', None)]),
    'Ingeniería Civil y Arquitectura': ('Practicante de obras, proyectos o diseño',
        [('Cementos Pacasmayo', 'cementospacasmayo.com.pe', 'https://pe.indeed.com/cmp/Cementos-Pacasmayo'),
         ('UNACEM', 'unacem.com.pe', None), ('AENZA', 'aenza.com', None), ('Cosapi', 'cosapi.com.pe', None)]),
    'Ingeniería Agroindustrial': ('Practicante de procesos, calidad o producción agrícola',
        [('Danper', 'danper.com', None), ('Camposol', 'camposol.com', 'https://pe.indeed.com/cmp/Camposol'),
         ('Virú', 'viru.com.pe', None), ('Grupo Gloria', 'grupogloria.com', None)]),
    'Derecho': ('Practicante legal o de asesoría jurídica',
        [('SUNAT', 'sunat.gob.pe', 'https://talentoperu.servir.gob.pe'),
         ('Poder Judicial', 'pj.gob.pe', 'https://talentoperu.servir.gob.pe'),
         ('Ministerio Público', 'gob.pe', 'https://talentoperu.servir.gob.pe'), ('BCP', 'viabcp.com', 'https://www.viabcp.com/unete-al-equipo-bcp/talento-joven')]),
    'Psicología': ('Practicante de recursos humanos y bienestar',
        [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus'), ('Alicorp', 'alicorp.com.pe', None),
         ('Telefónica', 'telefonica.com.pe', None), ('Ripley', 'ripley.com.pe', None)]),
    'Ciencias de la Salud': ('Practicante de salud, atención al paciente o gestión clínica',
        [('Clínica Internacional', 'clinicainternacional.com.pe', None), ('Sanna Clínicas', 'sanna.pe', None),
         ('EsSalud', 'essalud.gob.pe', None), ('MINSA', 'gob.pe', 'https://talentoperu.servir.gob.pe')])}
AFINIDAD = {'Programming_Skill': 'Ingeniería de Sistemas e IA', 'Communication_Skill': 'Marketing y Comunicación',
            'Teamwork': 'Administración y Negocios', 'GPA': 'Contabilidad y Finanzas',
            'Internship_Count': 'Administración y Negocios', 'Industry_Exposure': 'Ingeniería Industrial'}
MODALIDAD = {'High': 'Posiciones junior y programas trainee',
             'Medium': 'Prácticas preprofesionales y puestos de entrada',
             'Low': 'Prácticas de formación y programas de capacitación'}


def limpiar(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', str(texto).lower())
                   if c.isalnum() and not unicodedata.combining(c))


COLUMNAS = {'GPA': 'Promedio', 'Programming_Skill': 'Programación', 'Communication_Skill': 'Comunicación',
            'Teamwork': 'Trabajo en equipo', 'Internship_Count': 'Prácticas', 'Industry_Exposure': 'Exposición laboral'}
ALIAS = {limpiar(n): v for v in VARIABLES for n in (v, ETIQUETAS[v], CORTAS[v], COLUMNAS[v])}
COLUMNAS_ID = ('estudiante', 'nombre', 'nombres', 'codigo', 'id', 'alumno')
MAX_FILAS = 500


def entrenar_red(a, y_a, b, y_b, semilla=42, paciencia=25, max_epocas=300):
    red = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                        learning_rate_init=0.001, alpha=0.001, batch_size=32,
                        max_iter=1, random_state=semilla)
    pesos = compute_sample_weight('balanced', y_a)
    mejor_f1, mejor_red, mejor_epoca, sin_mejora = -1, None, 0, 0
    for epoca in range(1, max_epocas + 1):
        red.partial_fit(a, y_a, classes=np.array([0, 1, 2]), sample_weight=pesos)
        f1 = f1_score(y_b, red.predict(b), average='macro')
        if f1 > mejor_f1 + 1e-4:
            mejor_f1, mejor_red, mejor_epoca, sin_mejora = f1, copy.deepcopy(red), epoca, 0
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            break
    return mejor_red, mejor_epoca


def entrenar(ruta_csv=CSV, semilla=42):
    df = pd.read_csv(ruta_csv)
    X = df[VARIABLES]
    y = df['Employment_Class'].map({c: i for i, c in enumerate(CLASES)})
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                random_state=semilla, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                              random_state=semilla, stratify=y_tmp)
    escalador = StandardScaler().fit(X_tr)
    a, b, c = (escalador.transform(d) for d in (X_tr, X_va, X_te))
    regresion = LogisticRegression(max_iter=2000, class_weight='balanced',
                                   random_state=semilla).fit(a, y_tr.to_numpy())
    red, epoca = entrenar_red(a, y_tr.to_numpy(), b, y_va.to_numpy(), semilla)
    metricas = {}
    for nombre, modelo in (('Red neuronal (MLP)', red), ('Regresión Logística', regresion)):
        pred = modelo.predict(c)
        metricas[nombre] = {'exactitud': float(accuracy_score(y_te, pred)),
                            'f1_macro': float(f1_score(y_te, pred, average='macro'))}
    sistema = {'escalador': escalador, 'red': red, 'regresion': regresion,
               'perfil_clases': df.groupby('Employment_Class')[VARIABLES].mean().reindex(CLASES),
               'desviacion': df[VARIABLES].std(), 'minimo': df[VARIABLES].min(),
               'maximo': df[VARIABLES].max(), 'metricas': metricas,
               'epoca': epoca, 'registros': len(df), 'sklearn': sklearn.__version__}
    return sistema


def guardar(sistema, ruta=MODELO):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sistema, ruta)


def cargar(ruta=MODELO):
    try:
        sistema = joblib.load(ruta)
        if sistema.get('sklearn') != sklearn.__version__:
            raise ValueError('versión distinta de scikit-learn')
        return sistema
    except Exception:
        sistema = entrenar()
        guardar(sistema, ruta)
        return sistema


def validar(datos):
    valores, errores = {}, []
    for v in VARIABLES:
        crudo = datos.get(v)
        try:
            x = float(str(crudo).replace(',', '.'))
        except (TypeError, ValueError):
            errores.append(f'{ETIQUETAS[v]}: ingrese un número.')
            continue
        a, b = LIMITES[v]
        if not a <= x <= b:
            errores.append(f'{ETIQUETAS[v]}: el valor debe estar entre {a} y {b}.')
        elif v == 'Internship_Count' and x != int(x):
            errores.append(f'{ETIQUETAS[v]}: debe ser un número entero.')
        else:
            valores[v] = x
    return valores, errores


def recomendar(sistema, valores, clase):
    media_alta = sistema['perfil_clases'].loc['High']
    peso = np.clip(sistema['regresion'].coef_[2], 0, None)
    filas = []
    for v, p in zip(VARIABLES, peso):
        brecha = (media_alta[v] - valores[v]) / sistema['desviacion'][v]
        if brecha > 0.1:
            filas.append((brecha * p, v, valores[v], float(media_alta[v])))
    filas.sort(reverse=True)
    return [{'variable': ETIQUETAS[v], 'actual': round(act, 2), 'objetivo': round(obj, 2),
             'consejo': CONSEJOS[v]} for _, v, act, obj in filas[:3]]

def eliminar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto.lower()) if not unicodedata.combining(c))


def buscar_carrera(texto):
    t = limpiar(texto or '')
    if not t:
        return ''
    return next((c for c in CARRERAS if t == limpiar(c) or t in limpiar(c)), '')


def ofertas(comparacion, clase, carrera=''):
    elegida = buscar_carrera(carrera)
    if not elegida:
        mejor = max(range(len(VARIABLES)), key=lambda i: comparacion[i]['n_estudiante'])
        elegida = AFINIDAD[VARIABLES[mejor]]
    puesto, empresas = CARRERAS[elegida]
    tarjetas = []
    for nombre, dominio, enlace in empresas:
        busqueda = enlace or 'https://pe.indeed.com/q-practicante-' + eliminar_tildes(nombre).replace(' ', '-') + '-empleos.html'
        tarjetas.append({'empresa': nombre, 'dominio': dominio, 'puesto': puesto,
                         'modalidad': MODALIDAD[clase], 'enlace': busqueda})
    return {'carrera': elegida, 'indicada': bool(buscar_carrera(carrera)), 'empresas': tarjetas}


def predecir(sistema, valores, carrera=''):
    fila = pd.DataFrame([[valores[v] for v in VARIABLES]], columns=VARIABLES)
    z = sistema['escalador'].transform(fila)
    prob = sistema['red'].predict_proba(z)[0]
    clase = CLASES[int(np.argmax(prob))]
    segunda = CLASES[int(sistema['regresion'].predict(z)[0])]
    fuera = [ETIQUETAS[v] for v in VARIABLES
             if not sistema['minimo'][v] <= valores[v] <= sistema['maximo'][v]]
    media = sistema['perfil_clases']
    minimo, maximo = sistema['minimo'], sistema['maximo']

    def norm(v, x):
        return round(float(np.clip((x - minimo[v]) / (maximo[v] - minimo[v]), 0, 1)), 4)

    comparacion = []
    for v in VARIABLES:
        fila_c = {'variable': ETIQUETAS[v], 'corta': CORTAS[v], 'estudiante': valores[v],
                  'n_estudiante': norm(v, valores[v])}
        for c in CLASES:
            fila_c[c] = round(float(media.loc[c, v]), 2)
            fila_c['n_' + c] = norm(v, media.loc[c, v])
        comparacion.append(fila_c)
    return {'clase': clase, 'nivel': NIVELES[clase],
            'probabilidades': {c: round(float(p), 4) for c, p in zip(CLASES, prob)},
            'segunda_opinion': segunda, 'coincide': segunda == clase,
            'recomendaciones': recomendar(sistema, valores, clase),
            'comparacion': comparacion, 'fuera_de_rango': fuera,
            'ofertas': ofertas(comparacion, clase, carrera)}


def leer_tabla(archivo, nombre):
    if nombre.lower().endswith('.csv'):
        tabla = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8-sig')
    elif nombre.lower().endswith(('.xlsx', '.xls')):
        tabla = pd.read_excel(archivo)
    else:
        raise ValueError('Formato no admitido. Suba un archivo .xlsx o .csv.')
    columnas = {}
    for c in tabla.columns:
        v = ALIAS.get(limpiar(c))
        if v and v not in columnas.values():
            columnas[c] = v
    faltan = [ETIQUETAS[v] for v in VARIABLES if v not in columnas.values()]
    if faltan:
        raise ValueError('Faltan columnas: ' + ', '.join(faltan) + '.')
    if len(tabla) == 0:
        raise ValueError('El archivo no tiene filas con datos.')
    if len(tabla) > MAX_FILAS:
        raise ValueError(f'El archivo supera el máximo de {MAX_FILAS} estudiantes.')
    return tabla, columnas


def evaluar_tabla(sistema, tabla, columnas):
    id_col = next((c for c in tabla.columns if limpiar(c) in COLUMNAS_ID), None)
    car_col = next((c for c in tabla.columns if limpiar(c) == 'carrera'), None)
    filas = []
    for i, (_, fila) in enumerate(tabla.iterrows(), 1):
        valores, errores = validar({v: fila[c] for c, v in columnas.items()})
        nombre = str(fila[id_col]) if id_col is not None and pd.notna(fila[id_col]) else f'Estudiante {i}'
        registro = {'estudiante': nombre, 'errores': errores, 'valores': valores,
                    'carrera': str(fila[car_col]) if car_col is not None and pd.notna(fila[car_col]) else ''}
        if not errores:
            r = predecir(sistema, valores)
            registro.update({'clase': r['clase'], 'nivel': r['nivel'],
                             'prob_alta': r['probabilidades']['High'],
                             'prioridad': r['recomendaciones'][0]['variable'] if r['recomendaciones'] else '',
                             'segunda_opinion': r['segunda_opinion'], 'coincide': r['coincide']})
        filas.append(registro)
    return filas


def resumen(sistema, ruta_csv=CSV):
    df = pd.read_csv(ruta_csv)
    conteo = df['Employment_Class'].value_counts()
    peso = sistema['regresion'].coef_[2]
    orden = [VARIABLES[i] for i in np.argsort(peso)[::-1]]
    estad = []
    for v in VARIABLES:
        fila = {'variable': v, 'etiqueta': ETIQUETAS[v], 'corta': CORTAS[v],
                'minimo': round(float(df[v].min()), 2), 'media': round(float(df[v].mean()), 2),
                'maximo': round(float(df[v].max()), 2)}
        for c in CLASES:
            fila[c] = round(float(sistema['perfil_clases'].loc[c, v]), 2)
        estad.append(fila)
    return {'total': len(df), 'distribucion': {c: int(conteo.get(c, 0)) for c in CLASES},
            'medias': {v: round(float(df[v].mean()), 2) for v in VARIABLES},
            'estadisticas': estad, 'clave': [CORTAS[v] for v in orden[:2]],
            'muestra': df[['Student_No'] + VARIABLES + ['Employment_Class']].head(8).to_dict('records')}


def plantilla():
    ejemplo = pd.DataFrame([
        ['Estudiante 1', 7.2, 4, 5, 6, 1, 3, 'Ingeniería de Sistemas e IA'],
        ['Estudiante 2', 9.1, 8, 8, 7, 3, 8, 'Administración y Negocios'],
        ['Estudiante 3', 6.4, 2, 3, 3, 0, 1, 'Derecho']],
        columns=['Estudiante'] + [COLUMNAS[v] for v in VARIABLES] + ['Carrera'])
    return ejemplo
