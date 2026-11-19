from functools import cache
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tsp.agm import agm_prim
from tsp.christofides import vertices_grau_impar
from tsp.emparelhamento import (
    emparelhamento_perfeito_minimo,
    emparelhamento_peso_maximo_cardinalidade_maxima,
)
from tsp.instancia import carregar_instancia

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))


def _custo(custos: np.ndarray, pares: list[tuple[int, int]]) -> int:
    return sum(int(custos[u, v]) for u, v in pares)


def _eh_perfeito(pares: list[tuple[int, int]], vertices: list[int]) -> bool:
    cobertos = [v for par in pares for v in par]
    return sorted(cobertos) == sorted(vertices)


def _minimo_forca_bruta(custos: np.ndarray) -> int:
    n = len(custos)

    @cache
    def melhor(livres: frozenset[int]) -> int:
        if not livres:
            return 0
        u = min(livres)
        resto = livres - {u}
        return min(int(custos[u, v]) + melhor(resto - {v}) for v in resto)

    return melhor(frozenset(range(n)))


def _minimo_networkx(custos: np.ndarray, vertices: list[int]) -> int:
    g = nx.Graph()
    for a, u in enumerate(vertices):
        for v in vertices[a + 1 :]:
            g.add_edge(u, v, weight=int(custos[u, v]))
    return _custo(custos, list(nx.min_weight_matching(g)))


def _matriz_aleatoria(rng: np.random.Generator, n: int, maximo: int) -> np.ndarray:
    m = rng.integers(1, maximo + 1, size=(n, n))
    m = np.triu(m, 1)
    return (m + m.T).astype(np.int64)


def test_exemplo_pequeno():
    # Os pares baratos (0,1) e (2,3) custam 2; qualquer outro par custa 10.
    custos = np.full((4, 4), 10, dtype=np.int64)
    np.fill_diagonal(custos, 0)
    custos[0, 1] = custos[1, 0] = custos[2, 3] = custos[3, 2] = 1
    assert emparelhamento_perfeito_minimo(custos, [0, 1, 2, 3]) == [(0, 1), (2, 3)]


def test_usa_rotulos_originais_dos_vertices():
    custos = np.array(
        [[0, 9, 9, 9], [9, 0, 9, 1], [9, 9, 0, 9], [9, 1, 9, 0]], dtype=np.int64
    )
    assert emparelhamento_perfeito_minimo(custos, [3, 1]) == [(1, 3)]


def test_casos_de_borda():
    custos = np.zeros((3, 3), dtype=np.int64)
    assert emparelhamento_perfeito_minimo(custos, []) == []
    with pytest.raises(ValueError):
        emparelhamento_perfeito_minimo(custos, [0, 1, 2])
    assert emparelhamento_peso_maximo_cardinalidade_maxima(3, []) == [-1, -1, -1]


def test_cardinalidade_maxima_prefere_mais_arestas():
    # Caminho 0-1-2-3 com a aresta do meio pesada: o peso máximo puro usaria
    # só (1,2); com cardinalidade máxima o resultado tem de ser {(0,1), (2,3)}.
    arestas = [(0, 1, 1), (1, 2, 10), (2, 3, 1)]
    assert emparelhamento_peso_maximo_cardinalidade_maxima(4, arestas) == [1, 0, 3, 2]


@pytest.mark.parametrize("semente", range(200))
def test_contra_forca_bruta(semente: int):
    rng = np.random.default_rng(semente)
    n = int(rng.choice([2, 4, 6, 8, 10]))
    custos = _matriz_aleatoria(rng, n, maximo=int(rng.choice([3, 20, 1000])))
    pares = emparelhamento_perfeito_minimo(custos, list(range(n)))
    assert _eh_perfeito(pares, list(range(n)))
    assert _custo(custos, pares) == _minimo_forca_bruta(custos)


@pytest.mark.parametrize("semente", range(100))
def test_contra_networkx_em_grafos_maiores(semente: int):
    rng = np.random.default_rng(1000 + semente)
    n = 2 * int(rng.integers(6, 31))
    # Pesos pequenos geram muitos empates e exercitam flores aninhadas.
    custos = _matriz_aleatoria(rng, n, maximo=int(rng.choice([2, 5, 100])))
    vertices = list(range(n))
    pares = emparelhamento_perfeito_minimo(custos, vertices)
    assert _eh_perfeito(pares, vertices)
    assert _custo(custos, pares) == _minimo_networkx(custos, vertices)


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_impares_da_agm_das_instancias(caminho: Path):
    custos = carregar_instancia(caminho).custos
    impares = vertices_grau_impar(agm_prim(custos))
    pares = emparelhamento_perfeito_minimo(custos, impares)
    assert _eh_perfeito(pares, impares)
    assert _custo(custos, pares) == _minimo_networkx(custos, impares)
