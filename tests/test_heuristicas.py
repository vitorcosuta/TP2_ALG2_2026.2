import math
from pathlib import Path

import numpy as np
import pytest

from tsp.branch_and_bound import branch_and_bound, penalidades_held_karp
from tsp.heuristicas import (
    ciclo_fecho_convexo,
    insercao_fecho_convexo,
    tour_vizinho_mais_proximo,
    trivial,
    vizinho_mais_proximo,
)
from tsp.instancia import carregar_instancia
from tsp.medicao import medir
from tsp.tour import custo_tour, eh_tour_valido

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))
HEURISTICAS = [trivial, vizinho_mais_proximo, insercao_fecho_convexo]


def _distancias(pontos: np.ndarray) -> np.ndarray:
    d = pontos[:, None, :] - pontos[None, :, :]
    return np.rint(np.hypot(d[..., 0], d[..., 1])).astype(np.int64)


def _aleatoria(semente: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semente)
    pontos = rng.random((int(rng.integers(1, 12)), 2)) * 1000
    return pontos, _distancias(pontos)


@pytest.mark.parametrize("heuristica", HEURISTICAS)
@pytest.mark.parametrize("semente", range(40))
def test_tour_valido_comeca_em_zero_e_nao_bate_o_otimo(heuristica, semente):
    _, c = _aleatoria(semente)
    custo, tour = heuristica(c)
    assert eh_tour_valido(tour, len(c))
    assert not tour or tour[0] == 0
    assert custo == custo_tour(c, tour)
    assert custo >= branch_and_bound(c).custo


@pytest.mark.parametrize("heuristica", HEURISTICAS)
def test_instancia_vazia_e_minusculas(heuristica):
    for n in range(4):
        c = _distancias(np.arange(2 * n, dtype=float).reshape(n, 2) * [1, 3])
        custo, tour = heuristica(c)
        assert eh_tour_valido(tour, n) and custo == custo_tour(c, tour)


def test_trivial():
    c = carregar_instancia(PASTA / "burma14.tsp").custos
    custo, tour = trivial(c)
    assert tour == list(range(14)) and custo == custo_tour(c, tour)


def test_vizinho_mais_proximo_em_exemplo_a_mao():
    #   0 —1— 1 —1— 2, e 3 longe de todos: 0 → 1 → 2 → 3.
    c = np.array([[0, 1, 2, 9], [1, 0, 1, 8], [2, 1, 0, 7], [9, 8, 7, 0]])
    assert tour_vizinho_mais_proximo(c) == [0, 1, 2, 3]
    assert tour_vizinho_mais_proximo(c, inicio=3) == [3, 2, 1, 0]


def test_vizinho_mais_proximo_desempata_pelo_menor_indice():
    c = np.array([[0, 5, 5], [5, 0, 5], [5, 5, 0]])
    assert tour_vizinho_mais_proximo(c) == [0, 1, 2]


def test_pontos_num_circulo_dao_o_tour_otimo():
    # Todos os pontos estão no fecho: o próprio fecho é o ótimo.
    angulos = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    pontos = np.column_stack([np.cos(angulos), np.sin(angulos)]) * 1000
    pontos = pontos[np.random.default_rng(0).permutation(12)]
    c = _distancias(pontos)
    custo, _ = insercao_fecho_convexo(c, pontos)
    assert custo == branch_and_bound(c).custo


def test_ponto_interior_entra_na_aresta_mais_proxima():
    pontos = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 10.0]])
    c = _distancias(pontos)
    _, tour = insercao_fecho_convexo(c, pontos)
    # O ponto 4 fica entre 0 e 1 (aresta de baixo do quadrado).
    i = tour.index(4)
    assert {tour[i - 1], tour[(i + 1) % 5]} == {0, 1}


def test_pontos_colineares_usam_o_par_mais_distante():
    pontos = np.array([[0, 0], [30, 0], [10, 0], [50, 0], [20, 0.0]])
    c = _distancias(pontos)
    assert sorted(ciclo_fecho_convexo(pontos, c)) == [0, 3]
    custo, tour = insercao_fecho_convexo(c, pontos)
    assert eh_tour_valido(tour, 5) and custo == 100  # ida e volta


def test_sem_pontos_usa_mds_e_recupera_a_geometria():
    # Em instância euclidiana o MDS reconstrói os pontos (a menos de rotação
    # e reflexão), então o fecho e o tour têm o mesmo custo.
    pontos = np.random.default_rng(7).random((30, 2)) * 1000
    c = _distancias(pontos)
    com_pontos, _ = insercao_fecho_convexo(c, pontos)
    medicao = medir(lambda m: insercao_fecho_convexo(c, None, m), repeticoes=1)
    assert medicao.resultado[0] == com_pontos
    assert "total/mds" in medicao.tempos_s


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_instancias_tsplib(caminho):
    inst = carregar_instancia(caminho)
    c = inst.custos
    resultados = {h.__name__: h(c) for h in (trivial, vizinho_mais_proximo)}
    resultados["fecho"] = insercao_fecho_convexo(c, inst.coordenadas)
    for custo, tour in resultados.values():
        assert eh_tour_valido(tour, len(c)) and custo == custo_tour(c, tour)
    # A inserção fica a ≤ 20% do limite de Held-Karp em todas (pior: brazil58,
    # ~15%). O trivial não serve de comparação: em dantzig42 a ordem do
    # arquivo já está a < 1% do ótimo.
    _, limite = penalidades_held_karp(c)
    assert resultados["fecho"][0] <= 1.2 * limite


@pytest.mark.parametrize(
    ("heuristica", "etapas"),
    [
        (trivial, {"construcao", "custo"}),
        (vizinho_mais_proximo, {"construcao", "custo"}),
        (insercao_fecho_convexo, {"mds", "fecho", "insercao", "custo"}),
    ],
    ids=["trivial", "vizinho", "fecho"],
)
def test_etapas_medidas(heuristica, etapas):
    c = carregar_instancia(PASTA / "gr17.tsp").custos
    medicao = medir(lambda m: heuristica(c, medidor=m), repeticoes=2)
    assert medicao.resultado == heuristica(c)
    assert set(medicao.tempos_s) == {"total"} | {f"total/{e}" for e in etapas}
