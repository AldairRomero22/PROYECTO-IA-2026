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
    'Ingeniería de Sistemas e IA': {
        'puesto': 'Practicante de Desarrollo y Datos',
        'funciones': 'Apoya al equipo en programar y probar aplicaciones, preparar y analizar datos y automatizar procesos con código.',
        'habilidades': 'Programación, lógica, bases de datos y trabajo en equipo',
        'empresas': [('BCP', 'viabcp.com', 'https://www.viabcp.com/unete-al-equipo-bcp/talento-joven', 'Banca'),
                     ('Interbank', 'interbank.pe', 'https://pe.indeed.com/cmp/Interbank', 'Banca y seguros'),
                     ('Globant', 'globant.com', None, 'Tecnología y software'),
                     ('NTT DATA', 'nttdata.com', None, 'Consultoría de TI')]},
    'Administración y Negocios': {
        'puesto': 'Practicante de Gestión Comercial y Operaciones',
        'funciones': 'Colabora en el seguimiento de ventas, la elaboración de reportes de gestión y la coordinación con proveedores y áreas internas.',
        'habilidades': 'Análisis, organización, Excel y comunicación',
        'empresas': [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus', 'Bebidas y consumo masivo'),
                     ('Alicorp', 'alicorp.com.pe', None, 'Alimentos y consumo masivo'),
                     ('Falabella', 'falabella.com.pe', None, 'Retail'),
                     ('Supermercados Peruanos', 'spsa.pe', None, 'Supermercados')]},
    'Contabilidad y Finanzas': {
        'puesto': 'Practicante Contable y Financiero',
        'funciones': 'Registra y concilia operaciones, apoya en el cierre contable, la elaboración de estados financieros y el análisis de costos.',
        'habilidades': 'Contabilidad, Excel avanzado, atención al detalle y ética',
        'empresas': [('BBVA', 'bbva.pe', None, 'Banca'), ('Scotiabank', 'scotiabank.com.pe', None, 'Banca'),
                     ('KPMG', 'kpmg.com', None, 'Auditoría y consultoría'), ('PwC', 'pwc.com', None, 'Auditoría y consultoría')]},
    'Marketing y Comunicación': {
        'puesto': 'Practicante de Marketing y Marca',
        'funciones': 'Apoya en campañas, contenido para redes, estudios de mercado y seguimiento de resultados de las acciones de marca.',
        'habilidades': 'Comunicación, creatividad, redes sociales y análisis de datos',
        'empresas': [('Belcorp', 'belcorp.biz', None, 'Cosmética y venta directa'), ('Unilever', 'unilever.com', None, 'Consumo masivo'),
                     ('Ripley', 'ripley.com.pe', None, 'Retail'), ('Entel Perú', 'entel.pe', None, 'Telecomunicaciones')]},
    'Ingeniería Industrial': {
        'puesto': 'Practicante de Producción, Calidad y Logística',
        'funciones': 'Participa en la mejora de procesos, el control de indicadores de calidad, la planificación de producción y el manejo de inventarios.',
        'habilidades': 'Mejora continua, análisis de procesos, Excel y trabajo en planta',
        'empresas': [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus', 'Bebidas'),
                     ('Cementos Pacasmayo', 'cementospacasmayo.com.pe', 'https://pe.indeed.com/cmp/Cementos-Pacasmayo', 'Cemento y construcción'),
                     ('Alicorp', 'alicorp.com.pe', None, 'Alimentos'), ('Grupo Gloria', 'grupogloria.com', None, 'Lácteos y alimentos')]},
    'Ingeniería Civil y Arquitectura': {
        'puesto': 'Practicante de Obras y Proyectos',
        'funciones': 'Apoya en el control de avance de obra, la revisión de planos y metrados, la supervisión en campo y la documentación del proyecto.',
        'habilidades': 'Lectura de planos, AutoCAD o Revit, presupuestos y trabajo en campo',
        'empresas': [('Cementos Pacasmayo', 'cementospacasmayo.com.pe', 'https://pe.indeed.com/cmp/Cementos-Pacasmayo', 'Cemento y construcción'),
                     ('UNACEM', 'unacem.com.pe', None, 'Cemento'), ('AENZA', 'aenza.com', None, 'Infraestructura'),
                     ('Cosapi', 'cosapi.com.pe', None, 'Construcción e ingeniería')]},
    'Ingeniería Agroindustrial': {
        'puesto': 'Practicante de Procesos y Calidad Agroindustrial',
        'funciones': 'Colabora en el control de calidad de la fruta y hortalizas, el seguimiento de procesos de planta y la trazabilidad de la producción.',
        'habilidades': 'Inocuidad alimentaria, control de calidad, procesos y trabajo en campo',
        'empresas': [('Danper', 'danper.com', None, 'Agroexportación'),
                     ('Camposol', 'camposol.com', 'https://pe.indeed.com/cmp/Camposol', 'Agroexportación'),
                     ('Virú', 'viru.com.pe', None, 'Conservas y agroindustria'), ('Grupo Gloria', 'grupogloria.com', None, 'Lácteos y azúcar')]},
    'Derecho': {
        'puesto': 'Practicante Legal',
        'funciones': 'Revisa expedientes y contratos, redacta escritos y borradores de documentos, y hace seguimiento de plazos y trámites.',
        'habilidades': 'Redacción jurídica, investigación normativa, reserva y orden',
        'empresas': [('SUNAT', 'sunat.gob.pe', 'https://talentoperu.servir.gob.pe', 'Entidad pública tributaria'),
                     ('Poder Judicial', 'pj.gob.pe', 'https://talentoperu.servir.gob.pe', 'Entidad pública'),
                     ('Ministerio Público', 'gob.pe', 'https://talentoperu.servir.gob.pe', 'Entidad pública'),
                     ('BCP', 'viabcp.com', 'https://www.viabcp.com/unete-al-equipo-bcp/talento-joven', 'Banca y asesoría legal')]},
    'Psicología': {
        'puesto': 'Practicante de Recursos Humanos y Bienestar',
        'funciones': 'Apoya en reclutamiento y selección, evaluaciones de personal, clima laboral y actividades de bienestar y capacitación.',
        'habilidades': 'Entrevistas, evaluación psicológica, empatía y comunicación',
        'empresas': [('Backus', 'backus.pe', 'https://pe.indeed.com/cmp/Backus', 'Bebidas y consumo masivo'),
                     ('Alicorp', 'alicorp.com.pe', None, 'Alimentos'), ('Telefónica', 'telefonica.com.pe', None, 'Telecomunicaciones'),
                     ('Ripley', 'ripley.com.pe', None, 'Retail')]},
    'Ciencias de la Salud': {
        'puesto': 'Practicante de Salud y Atención al Paciente',
        'funciones': 'Asiste en la atención y el seguimiento de pacientes, el registro de historias clínicas y el apoyo a los equipos asistenciales.',
        'habilidades': 'Trato humano, protocolos de bioseguridad, orden y trabajo en equipo',
        'empresas': [('Clínica Internacional', 'clinicainternacional.com.pe', None, 'Clínica privada'),
                     ('Sanna Clínicas', 'sanna.pe', None, 'Red de clínicas'),
                     ('EsSalud', 'essalud.gob.pe', None, 'Seguridad social de salud'),
                     ('MINSA', 'gob.pe', 'https://talentoperu.servir.gob.pe', 'Entidad pública de salud')]}}
ENFOQUE = {
    'Ingeniería de Sistemas e IA': ({}, {
        'Communication_Skill': 'Practicar cómo explicar ideas técnicas a personas no técnicas: documentación, demos y presentaciones.',
        'Teamwork': 'Trabajar en proyectos en equipo con Git y metodologías ágiles, o participar en hackatones.',
        'Industry_Exposure': 'Acercarse al sector tecnológico: meetups, hackatones, charlas con empresas y mentorías con desarrolladores.',
        'Internship_Count': 'Buscar prácticas en desarrollo de software, datos o infraestructura TI.'}),
    'Administración y Negocios': ({'Programming_Skill': 0.4}, {
        'Programming_Skill': 'Dominar herramientas digitales de gestión: Excel avanzado, Power BI y manejo de datos para decidir mejor.',
        'Communication_Skill': 'Fortalecer la comunicación con clientes y equipos: presentaciones, negociación e informes ejecutivos.',
        'Industry_Exposure': 'Acercarse a empresas: ferias de empleo, visitas a áreas comerciales y mentorías con jefes de área.',
        'Internship_Count': 'Buscar prácticas en áreas comerciales, de operaciones o administrativas.'}),
    'Contabilidad y Finanzas': ({'Programming_Skill': 0.4, 'GPA': 1.2}, {
        'GPA': 'Reforzar contabilidad financiera, costos y tributación: repasar los cursos con menor nota y practicar con casos reales.',
        'Programming_Skill': 'Manejar herramientas de análisis: Excel avanzado, Power BI y sistemas contables o ERP.',
        'Communication_Skill': 'Practicar la presentación clara de informes y estados financieros a personas que no son contadoras.',
        'Industry_Exposure': 'Acercarse a estudios contables, áreas financieras y bancos: ferias, charlas y mentorías con contadores.',
        'Internship_Count': 'Buscar prácticas contables, de tesorería o de auditoría.'}),
    'Marketing y Comunicación': ({'Programming_Skill': 0.3, 'Communication_Skill': 1.3}, {
        'Programming_Skill': 'Aprender analítica digital: Google Analytics, Power BI y automatización de campañas para medir resultados de marca.',
        'Communication_Skill': 'Practicar redacción publicitaria, storytelling y presentación de propuestas de campaña a clientes.',
        'Teamwork': 'Participar en proyectos creativos grupales, agencias universitarias o concursos de marketing.',
        'Industry_Exposure': 'Acercarse a agencias y áreas de marca: ferias, charlas y mentorías con profesionales de marketing.',
        'Internship_Count': 'Buscar prácticas en marketing digital, redes sociales o investigación de mercados.'}),
    'Ingeniería Industrial': ({'Programming_Skill': 0.6, 'Industry_Exposure': 1.2}, {
        'Programming_Skill': 'Usar herramientas de análisis y automatización: Excel avanzado, Power BI, Python básico y simulación de procesos.',
        'Communication_Skill': 'Practicar la presentación de indicadores y propuestas de mejora a jefaturas de planta.',
        'Teamwork': 'Participar en proyectos de mejora continua con equipos multidisciplinarios en planta.',
        'Industry_Exposure': 'Visitar plantas y almacenes: ferias, visitas guiadas y mentorías con ingenieros de producción y logística.',
        'Internship_Count': 'Buscar prácticas en producción, calidad o logística.'}),
    'Ingeniería Civil y Arquitectura': ({'Programming_Skill': 0.5, 'Industry_Exposure': 1.3}, {
        'Programming_Skill': 'Dominar el software de la carrera: AutoCAD, Revit, S10 o MS Project para planos, presupuestos y cronogramas.',
        'Communication_Skill': 'Practicar la redacción de informes técnicos y la comunicación con contratistas y supervisores.',
        'Teamwork': 'Trabajar en equipos de obra y en proyectos grupales con roles definidos.',
        'Industry_Exposure': 'Acercarse a obras: visitas de campo, ferias y mentorías con residentes e ingenieros de proyecto.',
        'Internship_Count': 'Buscar prácticas en obra, supervisión o gestión de proyectos.'}),
    'Ingeniería Agroindustrial': ({'Programming_Skill': 0.4, 'Industry_Exposure': 1.3}, {
        'Programming_Skill': 'Aprender herramientas de datos para el campo y la planta: Excel avanzado, Power BI y sistemas de trazabilidad y calidad.',
        'Communication_Skill': 'Practicar la comunicación clara de resultados de calidad y producción a supervisores y equipos de campo.',
        'Teamwork': 'Trabajar en equipos de campaña y de planta, donde la coordinación es clave.',
        'Industry_Exposure': 'Visitar fundos y plantas de proceso: ferias, visitas guiadas y mentorías con ingenieros agroindustriales.',
        'Internship_Count': 'Buscar prácticas en campo, planta de proceso o aseguramiento de calidad.'}),
    'Derecho': ({'Programming_Skill': 0.1, 'Communication_Skill': 1.4, 'GPA': 1.2}, {
        'GPA': 'Reforzar los cursos jurídicos con menor nota y practicar con casos y jurisprudencia.',
        'Programming_Skill': 'Familiarizarse con herramientas legales digitales: buscadores de jurisprudencia, expediente electrónico y ofimática avanzada.',
        'Communication_Skill': 'Entrenar oratoria, argumentación y redacción jurídica: debates, simulacros de audiencia y escritos.',
        'Teamwork': 'Participar en clínicas jurídicas y trabajos en equipo con casos reales.',
        'Industry_Exposure': 'Acercarse a estudios jurídicos, entidades públicas y áreas legales: charlas, ferias y mentorías con abogados.',
        'Internship_Count': 'Buscar prácticas en estudios jurídicos, juzgados o áreas legales de empresas.'}),
    'Psicología': ({'Programming_Skill': 0.1, 'Communication_Skill': 1.3, 'Teamwork': 1.1}, {
        'Programming_Skill': 'Aprender herramientas digitales de evaluación: Excel, SPSS o plataformas de pruebas psicométricas.',
        'Communication_Skill': 'Entrenar la escucha activa, la entrevista y la comunicación empática.',
        'Teamwork': 'Participar en talleres y proyectos grupales de intervención y bienestar.',
        'Industry_Exposure': 'Acercarse a áreas de recursos humanos y consultoras: charlas, ferias y mentorías con psicólogos organizacionales.',
        'Internship_Count': 'Buscar prácticas en recursos humanos, selección o bienestar.'}),
    'Ciencias de la Salud': ({'Programming_Skill': 0.1, 'Communication_Skill': 1.2, 'Teamwork': 1.2}, {
        'Programming_Skill': 'Familiarizarse con herramientas digitales clínicas: historias clínicas electrónicas y ofimática básica.',
        'Communication_Skill': 'Practicar el trato con el paciente y la comunicación clara con familiares y equipo de salud.',
        'Teamwork': 'Trabajar en equipos multidisciplinarios, como en prácticas hospitalarias y campañas de salud.',
        'Industry_Exposure': 'Acercarse a clínicas y hospitales: voluntariados, campañas de salud y mentorías con profesionales.',
        'Internship_Count': 'Buscar internados, voluntariados o prácticas en centros de salud y clínicas.'})}
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


def recomendar(sistema, valores, clase, carrera=''):
    media_alta = sistema['perfil_clases'].loc['High']
    peso = np.clip(sistema['regresion'].coef_[2], 0, None)
    relevancia, consejos = ENFOQUE.get(carrera, ({}, {}))
    filas = []
    for v, p in zip(VARIABLES, peso):
        brecha = (media_alta[v] - valores[v]) / sistema['desviacion'][v]
        if brecha > 0.1:
            filas.append((brecha * p * relevancia.get(v, 1.0), v, valores[v], float(media_alta[v])))
    filas.sort(reverse=True)
    digital = v_digital(carrera)
    return [{'variable': digital if v == 'Programming_Skill' and digital else ETIQUETAS[v],
             'actual': round(act, 2), 'objetivo': round(obj, 2),
             'consejo': consejos.get(v, CONSEJOS[v])} for _, v, act, obj in filas[:3]]


def v_digital(carrera):
    return '' if carrera in ('', 'Ingeniería de Sistemas e IA') else 'Herramientas digitales (programación)'


def eliminar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto.lower()) if not unicodedata.combining(c))


def buscar_carrera(texto):
    t = limpiar(texto or '')
    if not t:
        return ''
    return next((c for c in CARRERAS if t == limpiar(c) or t in limpiar(c)), '')


def resolver_carrera(comparacion, carrera):
    elegida = buscar_carrera(carrera)
    if elegida:
        return elegida, True
    mejor = max(range(len(VARIABLES)), key=lambda i: comparacion[i]['n_estudiante'])
    return AFINIDAD[VARIABLES[mejor]], False


def ofertas(comparacion, clase, carrera=''):
    elegida, indicada = resolver_carrera(comparacion, carrera)
    datos = CARRERAS[elegida]
    tarjetas = []
    for nombre, dominio, enlace, rubro in datos['empresas']:
        busqueda = enlace or 'https://pe.indeed.com/q-practicante-' + eliminar_tildes(nombre).replace(' ', '-') + '-empleos.html'
        tarjetas.append({'empresa': nombre, 'dominio': dominio, 'rubro': rubro, 'puesto': datos['puesto'],
                         'funciones': datos['funciones'], 'habilidades': datos['habilidades'],
                         'modalidad': MODALIDAD[clase], 'enlace': busqueda})
    return {'carrera': elegida, 'indicada': indicada, 'empresas': tarjetas}


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
            'recomendaciones': recomendar(sistema, valores, clase, resolver_carrera(comparacion, carrera)[0]),
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
