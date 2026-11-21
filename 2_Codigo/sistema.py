import copy
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
CONSEJOS = {'GPA': 'Reforzar el rendimiento académico: tutorías, grupos de estudio y repaso de los cursos con menor nota.',
            'Programming_Skill': 'Practicar programación de forma constante con proyectos propios, retos en línea y cursos de especialización.',
            'Communication_Skill': 'Entrenar la comunicación oral y escrita: exposiciones, talleres de oratoria y redacción de informes.',
            'Teamwork': 'Participar en proyectos grupales, hackatones o voluntariados para fortalecer el trabajo colaborativo.',
            'Internship_Count': 'Buscar prácticas preprofesionales o pasantías, aunque sean de corta duración.',
            'Industry_Exposure': 'Acercarse al entorno laboral: charlas con empresas, ferias de empleo, visitas y mentorías con profesionales.'}


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


def predecir(sistema, valores):
    fila = pd.DataFrame([[valores[v] for v in VARIABLES]], columns=VARIABLES)
    z = sistema['escalador'].transform(fila)
    prob = sistema['red'].predict_proba(z)[0]
    clase = CLASES[int(np.argmax(prob))]
    segunda = CLASES[int(sistema['regresion'].predict(z)[0])]
    fuera = [ETIQUETAS[v] for v in VARIABLES
             if not sistema['minimo'][v] <= valores[v] <= sistema['maximo'][v]]
    media = sistema['perfil_clases']
    return {'clase': clase, 'nivel': NIVELES[clase],
            'probabilidades': {c: round(float(p), 4) for c, p in zip(CLASES, prob)},
            'segunda_opinion': segunda, 'coincide': segunda == clase,
            'recomendaciones': recomendar(sistema, valores, clase),
            'comparacion': [{'variable': ETIQUETAS[v], 'estudiante': valores[v],
                             'Low': round(float(media.loc['Low', v]), 2),
                             'Medium': round(float(media.loc['Medium', v]), 2),
                             'High': round(float(media.loc['High', v]), 2)} for v in VARIABLES],
            'fuera_de_rango': fuera}
