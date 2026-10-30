"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def _recorrer_red(network, variables_pendientes: list, eventos_pasados: dict)->float:
    
    if not variables_pendientes:
        return 1
    
    elif variables_pendientes[0] in eventos_pasados:
        y = variables_pendientes[0]
        nuevos_eventos = {**eventos_pasados, y: eventos_pasados[y]}
        return (
            network.probability(y, eventos_pasados[y], eventos_pasados) * _recorrer_red(network, variables_pendientes[1:], nuevos_eventos)
            )

    else:
        p_y = 0
        y = variables_pendientes[0]
        
        for valor in (0,1):
            nuevos_eventos = {**eventos_pasados, y: valor}
            p_y += (
                network.probability(y, valor, eventos_pasados) * _recorrer_red(network, variables_pendientes[1:], nuevos_eventos)
                )
        
        return p_y

def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    pesos = {}
    for valor in (0,1):

        # verificar si la consulta ya está observada y si el valor actual lo contradice; saltar de una vez
        if query in evidence and evidence[query] != valor:
          pesos[valor] = 0
        else:
          # une el diccionario de evidencia con la variable consulta fijada en este momento en un solo diccionario
          eventos_pasados = {**evidence, query: valor}
          pesos[valor] = _recorrer_red(network, network.variables[:], eventos_pasados)

    return normalize(pesos)
