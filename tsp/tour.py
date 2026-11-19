"""Representação e custo de tours, compartilhados por todos os algoritmos.

Convenção: um tour é uma lista com os n vértices (base 0), cada um exatamente
uma vez, sem repetir o vértice inicial no fim. A aresta de volta do último
vértice ao primeiro está implícita e entra no custo.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from tsp.instancia import MatrizCustos

Tour = list[int]


def eh_tour_valido(tour: Sequence[int], n: int) -> bool:
    """Indica se ``tour`` visita cada um dos n vértices exatamente uma vez."""
    return len(tour) == n and sorted(tour) == list(range(n))


def custo_tour(custos: MatrizCustos, tour: Sequence[int]) -> int:
    """Custo do ciclo, incluindo a aresta de volta ao vértice inicial."""
    if len(tour) < 2:
        return 0
    custo = sum(int(custos[u, v]) for u, v in pairwise(tour))
    return custo + int(custos[tour[-1], tour[0]])
