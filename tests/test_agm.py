from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from tsp.agm import agm_prim, arestas_agm, custo_agm, filhos_agm, graus_agm
from tsp.instancia import carregar_instancia

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))


def _eh_arvore_geradora(pai: np.ndarray, raiz: int) -> bool:
    """Todo vértice chega à raiz seguindo ``pai`` sem ciclos."""
    n = len(pai)
    if pai[raiz] != -1 or (pai < 0).sum() != 1:
        return False
    for v in range(n):
        passos = 0
        while v != raiz:
            v = int(pai[v])
            passos += 1
            if passos > n:
                return False
    return True


def test_exemplo_pequeno():
    # 0-1 (1), 1-2 (2), 2-3 (1) formam a AGM de custo 4.
    custos = np.array(
        [[0, 1, 5, 6], [1, 0, 2, 7], [5, 2, 0, 1], [6, 7, 1, 0]], dtype=np.int64
    )
    copia = custos.copy()
    pai = agm_prim(custos)
    assert pai.tolist() == [-1, 0, 1, 2]
    assert custo_agm(custos, pai) == 4
    assert sorted(arestas_agm(pai)) == [(0, 1), (1, 2), (2, 3)]
    assert graus_agm(pai).tolist() == [1, 2, 2, 1]
    assert filhos_agm(pai) == [[1], [2], [3], []]
    assert (custos == copia).all()


def test_grafo_com_um_e_dois_vertices():
    assert agm_prim(np.zeros((1, 1), dtype=np.int64)).tolist() == [-1]
    pai = agm_prim(np.array([[0, 3], [3, 0]], dtype=np.int64))
    assert pai.tolist() == [-1, 0]


def test_raiz_invalida():
    with pytest.raises(ValueError):
        agm_prim(np.zeros((3, 3), dtype=np.int64), raiz=3)


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_agm_das_instancias(caminho: Path):
    custos = carregar_instancia(caminho).custos
    n = len(custos)
    esperado = int(minimum_spanning_tree(custos).sum())
    for raiz in (0, n - 1):
        pai = agm_prim(custos, raiz)
        assert _eh_arvore_geradora(pai, raiz)
        assert len(arestas_agm(pai)) == n - 1
        assert graus_agm(pai).sum() == 2 * (n - 1)
        assert custo_agm(custos, pai) == esperado
