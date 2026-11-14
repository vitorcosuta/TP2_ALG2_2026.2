from pathlib import Path

import numpy as np
import pytest

from tsp.instancia import carregar_instancia, fechamento_metrico

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))

# Custo ótimo publicado na TSPLIB (pesos originais, sem fecho métrico).
OTIMOS_TSPLIB = {
    "att48": 10628,
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "eil51": 426,
    "fri26": 937,
    "gr24": 1272,
    "gr48": 5046,
    "st70": 675,
    "ulysses16": 6859,
    "ulysses22": 7013,
}


def _ler_tour(caminho: Path) -> list[int]:
    tokens = caminho.read_text().split()
    inicio = tokens.index("TOUR_SECTION") + 1
    tour = []
    for token in tokens[inicio:]:
        if token == "-1" or token == "EOF":
            break
        tour.append(int(token) - 1)
    return tour


def _custo(custos: np.ndarray, tour: list[int]) -> int:
    return int(
        sum(custos[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    )


def test_fechamento_corrige_triangulo():
    custos = np.array([[0, 1, 10], [1, 0, 2], [10, 2, 0]], dtype=np.int64)
    esperado = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=np.int64)
    assert (fechamento_metrico(custos) == esperado).all()
    assert custos[0, 2] == 10  # entrada intacta


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_custos_sao_metricos(caminho: Path):
    d = carregar_instancia(caminho).custos
    fora_diagonal = ~np.eye(len(d), dtype=bool)
    assert (d >= 0).all()
    assert (np.diag(d) == 0).all()
    assert (d[fora_diagonal] > 0).all()
    assert (d == d.T).all()
    melhor_desvio = (d[:, :, None] + d[None, :, :]).min(axis=1)
    assert (d <= melhor_desvio).all()


@pytest.mark.parametrize("nome", sorted(OTIMOS_TSPLIB))
def test_leitura_reproduz_otimo_tsplib(nome: str):
    instancia = carregar_instancia(PASTA / f"{nome}.tsp", metrica=False)
    tour = _ler_tour(PASTA / f"{nome}.opt.tour")
    assert sorted(tour) == list(range(instancia.dimensao))
    assert _custo(instancia.custos, tour) == OTIMOS_TSPLIB[nome]


ORIGEM_COORDENADAS = {
    "bayg29": "DISPLAY_DATA_SECTION",
    "bays29": "DISPLAY_DATA_SECTION",
    "dantzig42": "DISPLAY_DATA_SECTION",
    "att48": "NODE_COORD_SECTION",
    "berlin52": "NODE_COORD_SECTION",
    "burma14": "NODE_COORD_SECTION",
    "eil51": "NODE_COORD_SECTION",
    "st70": "NODE_COORD_SECTION",
    "ulysses16": "NODE_COORD_SECTION",
    "ulysses22": "NODE_COORD_SECTION",
}


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_origem_das_coordenadas(caminho: Path):
    instancia = carregar_instancia(caminho)
    origem = ORIGEM_COORDENADAS.get(caminho.stem)
    assert instancia.origem_coordenadas == origem
    if origem is None:
        assert instancia.coordenadas is None
    else:
        assert instancia.coordenadas is not None
        assert instancia.coordenadas.shape == (instancia.dimensao, 2)
        assert np.isfinite(instancia.coordenadas).all()
