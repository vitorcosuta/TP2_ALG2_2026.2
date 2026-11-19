from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tsp.agm import agm_prim, arestas_agm, custo_agm, graus_agm
from tsp.christofides import christofides, circuito_euleriano, vertices_grau_impar
from tsp.emparelhamento import emparelhamento_perfeito_minimo
from tsp.instancia import carregar_instancia
from tsp.tour import custo_tour, eh_tour_valido

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))


def _ler_tour(caminho: Path) -> list[int]:
    tokens = caminho.read_text().split()
    tour = []
    for token in tokens[tokens.index("TOUR_SECTION") + 1 :]:
        if token in ("-1", "EOF"):
            break
        tour.append(int(token) - 1)
    return tour


def _multiconjunto(arestas) -> Counter:
    return Counter((min(u, v), max(u, v)) for u, v in arestas)


def test_vertices_grau_impar():
    # Estrela com centro 0 e três folhas: todos têm grau ímpar (3, 1, 1, 1).
    pai = np.array([-1, 0, 0, 0], dtype=np.int64)
    assert vertices_grau_impar(pai) == [0, 1, 2, 3]
    # Caminho 0-1-2: só as pontas têm grau ímpar.
    assert vertices_grau_impar(np.array([-1, 0, 1], dtype=np.int64)) == [0, 2]


def test_circuito_euleriano_com_arestas_paralelas():
    # Triângulo 0-1-2 mais uma aresta 0-1 repetida e outra 1-2... grau par:
    arestas = [(0, 1), (1, 2), (2, 0), (0, 1), (1, 0)]
    circuito = circuito_euleriano(3, arestas)
    assert circuito[0] == circuito[-1] == 0
    assert len(circuito) == len(arestas) + 1
    assert _multiconjunto(pairwise(circuito)) == _multiconjunto(arestas)


def test_exemplo_do_shortcutting():
    # O exemplo a-b-c-b-d (+ volta) vira a-b-c-d.
    from tsp.twice_around_the_tree import remover_repetidos

    assert remover_repetidos([0, 1, 2, 1, 3, 0]) == [0, 1, 2, 3]


def test_grafos_minimos():
    assert christofides(np.zeros((1, 1), dtype=np.int64)) == (0, [0])
    custos = np.array([[0, 3], [3, 0]], dtype=np.int64)
    assert christofides(custos) == (6, [0, 1])


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_instancias(caminho: Path):
    custos = carregar_instancia(caminho).custos
    n = len(custos)
    pai = agm_prim(custos)
    impares = vertices_grau_impar(pai)
    assert len(impares) % 2 == 0
    emparelhamento = emparelhamento_perfeito_minimo(custos, impares)

    # A união AGM + M é euleriana e o circuito usa cada aresta uma vez.
    multigrafo = arestas_agm(pai) + emparelhamento
    graus = graus_agm(pai)
    for u, v in emparelhamento:
        graus[u] += 1
        graus[v] += 1
    assert (graus % 2 == 0).all()
    circuito = circuito_euleriano(n, multigrafo)
    assert _multiconjunto(pairwise(circuito)) == _multiconjunto(multigrafo)

    custo, tour = christofides(custos)
    assert eh_tour_valido(tour, n)
    assert tour[0] == 0
    assert custo == custo_tour(custos, tour)
    custo_m = sum(int(custos[u, v]) for u, v in emparelhamento)
    assert custo <= custo_agm(custos, pai) + custo_m

    # Garantia de 1,5: o ótimo métrico é ≤ custo do tour ótimo da TSPLIB.
    otimo = caminho.with_suffix(".opt.tour")
    if otimo.exists():
        assert 2 * custo <= 3 * custo_tour(custos, _ler_tour(otimo))
