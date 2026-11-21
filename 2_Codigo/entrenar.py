import time

import sistema


if __name__ == '__main__':
    t0 = time.perf_counter()
    modelo = sistema.entrenar()
    sistema.guardar(modelo)
    print(f'Entrenamiento completado en {time.perf_counter() - t0:.1f} s '
          f'({modelo["registros"]} registros, mejor época {modelo["epoca"]})')
    for nombre, m in modelo['metricas'].items():
        print(f'{nombre:<22} exactitud = {m["exactitud"]:.4f} | F1-macro = {m["f1_macro"]:.4f}')
    print(f'Sistema guardado en {sistema.MODELO}')
