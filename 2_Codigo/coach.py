import json
import os
import time
import urllib.request
from collections import defaultdict, deque

URL = 'https://api.anthropic.com/v1/messages'
MODELO_POR_DEFECTO = 'claude-haiku-5-5'
LIMITE_POR_IP = 5
VENTANA = 3600
LIMITE_DIARIO = int(os.environ.get('COACH_LIMITE_DIARIO', '150'))
SISTEMA = ('Eres un orientador de carrera para estudiantes universitarios de Perú. '
           'Responde siempre en español, con un tono cercano y motivador, y solo con un JSON válido, sin texto adicional. '
           'Propón acciones concretas, realistas y gratuitas o de bajo costo. '
           'No inventes empresas, cifras ni convocatorias, y no prometas que conseguirá empleo.')
FORMATO = ('Devuelve exactamente este JSON: {"resumen": "2 a 3 oraciones sobre su situación y su mayor fortaleza", '
           '"semanas": [{"foco": "tema de la semana", "acciones": ["acción concreta", "acción concreta"]}], '
           '"cierre": "una frase final de ánimo"}. Incluye exactamente 4 semanas con 2 o 3 acciones cada una, '
           'priorizando las variables más débiles y adaptadas a la carrera.')

_por_ip = defaultdict(deque)
_dia = {'fecha': '', 'usos': 0}


def disponible():
    return bool(os.environ.get('ANTHROPIC_API_KEY'))


def permitir(ip):
    ahora = time.time()
    cola = _por_ip[ip]
    while cola and ahora - cola[0] > VENTANA:
        cola.popleft()
    hoy = time.strftime('%Y-%m-%d')
    if _dia['fecha'] != hoy:
        _dia.update(fecha=hoy, usos=0)
    if len(cola) >= LIMITE_POR_IP or _dia['usos'] >= LIMITE_DIARIO:
        return False
    cola.append(ahora)
    _dia['usos'] += 1
    return True


def instrucciones(resultado, carrera):
    variables = '\n'.join(f"- {f['variable']}: {f['estudiante']} (promedio de la clase Alta: {f['High']})"
                          for f in resultado['comparacion'])
    aportes = ', '.join(f"{e['variable']} {e['aporte']:+}" for e in resultado['explicacion'])
    return (f"Carrera: {carrera}\nNivel de competitividad laboral: {resultado['nivel']} "
            f"(probabilidad de Alta: {resultado['probabilidades']['High'] * 100:.1f} %)\n"
            f"Variables del estudiante:\n{variables}\n"
            f"Aporte de cada variable a la probabilidad de Alta: {aportes}\n\n{FORMATO}")


def llamar(texto):
    cuerpo = json.dumps({'model': os.environ.get('ANTHROPIC_MODEL', MODELO_POR_DEFECTO), 'max_tokens': 900,
                         'system': SISTEMA, 'messages': [{'role': 'user', 'content': texto}]}).encode()
    pedido = urllib.request.Request(URL, data=cuerpo, headers={
        'content-type': 'application/json', 'anthropic-version': '2023-06-01',
        'x-api-key': os.environ['ANTHROPIC_API_KEY']})
    with urllib.request.urlopen(pedido, timeout=30) as respuesta:
        return json.load(respuesta)['content'][0]['text']


def generar(resultado, carrera):
    texto = llamar(instrucciones(resultado, carrera))
    datos = json.loads(texto[texto.index('{'):texto.rindex('}') + 1])
    semanas = [{'foco': str(s['foco'])[:80], 'acciones': [str(a)[:240] for a in s['acciones'][:3]]}
               for s in datos['semanas'][:4]]
    if not semanas or not all(s['acciones'] for s in semanas):
        raise ValueError('Respuesta incompleta')
    return {'resumen': str(datos['resumen'])[:600], 'semanas': semanas, 'cierre': str(datos.get('cierre', ''))[:240]}
