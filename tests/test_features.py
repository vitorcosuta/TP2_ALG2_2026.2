import math
from pathlib import Path

import numpy as np
import pytest

from tsp.features import (
    entropia_espacial,
    extrair_features,
    features_agm,
    features_distancias,
    fracao_fecho_convexo,
    indice_clark_evans,
    mds_classico,
    razao_aspecto,
)
from tsp.instancia import carregar_instancia

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))


def _grade(lado: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(lado), np.arange(lado))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def _distancias(pontos: np.ndarray) -> np.ndarray:
    d = pontos[:, None, :] - pontos[None, :, :]
    return np.rint(np.hypot(d[..., 0], d[..., 1])).astype(np.int64)


def test_razao_aspecto():
    assert razao_aspecto(_grade(4)) == 1.0
    faixa = np.array([[0, 0], [10, 1], [5, 0.5]], dtype=float)
    assert razao_aspecto(faixa) == pytest.approx(0.1)
    assert math.isnan(razao_aspecto(np.zeros((3, 2))))


def test_fracao_fecho_convexo():
    # Quadrado com um ponto no centro: 4 de 5 no fecho.
    pontos = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1]], dtype=float)
    assert fracao_fecho_convexo(pontos) == pytest.approx(0.8)
    # Círculo: todos no fecho.
    t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    assert fracao_fecho_convexo(np.column_stack([np.cos(t), np.sin(t)])) == 1.0
    # Colineares: o fecho não existe.
    assert math.isnan(fracao_fecho_convexo(np.array([[0, 0], [1, 1], [2, 2.0]])))


def test_entropia_espalhado_maior_que_agrupado():
    rng = np.random.default_rng(0)
    uniforme = rng.random((200, 2))
    clusters = np.vstack([rng.normal(c, 0.01, (100, 2)) for c in ((0, 0), (1, 1))])
    assert entropia_espacial(_grade(10), celulas_por_lado=5) == pytest.approx(1.0)
    assert entropia_espacial(uniforme) > 0.9
    assert entropia_espacial(clusters) < 0.3


def test_clark_evans_distingue_padroes():
    rng = np.random.default_rng(1)
    grade = _grade(10)
    aleatorio = rng.random((300, 2))
    clusters = np.vstack(
        [rng.normal(c, 0.02, (50, 2)) for c in rng.random((4, 2)) * 10]
    )
    assert indice_clark_evans(grade) > 1.5  # regular
    assert 0.8 < indice_clark_evans(aleatorio) < 1.2  # ~Poisson
    assert indice_clark_evans(clusters) < 0.5  # agrupado


def test_features_agm_estrela_e_caminho():
    # Estrela: centro 0 a distância 1 de todos; folhas a distância 2 entre si.
    n = 6
    estrela = np.full((n, n), 2, dtype=np.int64)
    estrela[0, :] = estrela[:, 0] = 1
    np.fill_diagonal(estrela, 0)
    f = features_agm(estrela)
    assert f["grau_max_agm"] == n - 1
    assert f["fracao_folhas_agm"] == pytest.approx((n - 1) / n)
    assert f["cv_arestas_agm"] == 0.0
    # Pontos numa reta: a AGM é um caminho.
    caminho = _distancias(np.column_stack([np.arange(n) * 3.0, np.zeros(n)]))
    f = features_agm(caminho)
    assert f["grau_max_agm"] == 2 and f["fracao_impares_agm"] == pytest.approx(2 / n)


def test_cv_vizinho_mais_proximo():
    assert (
        features_distancias(_distancias(_grade(5) * 10))["cv_vizinho_mais_proximo"]
        == 0.0
    )
    # Dois pontos colados e um longe: vizinhos a 1, 1 e ~100.
    pontos = np.array([[0, 0], [1, 0], [100, 0]], dtype=float)
    assert features_distancias(_distancias(pontos))["cv_vizinho_mais_proximo"] > 1


def test_mds_recupera_distancias_euclidianas():
    rng = np.random.default_rng(2)
    pontos = rng.random((30, 2)) * 1000
    custos = _distancias(pontos)
    reconstruidos, qualidade = mds_classico(custos)
    assert qualidade > 0.99
    erro = np.abs(_distancias(reconstruidos) - custos).max()
    assert erro <= 2  # só o arredondamento
    # As features invariantes a rotação e translação batem.
    assert fracao_fecho_convexo(reconstruidos) == fracao_fecho_convexo(pontos)


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
@pytest.mark.parametrize("usar_mds", [False, True])
def test_instancias(caminho: Path, usar_mds: bool):
    instancia = carregar_instancia(caminho)
    linha = extrair_features(instancia, usar_mds=usar_mds)
    assert linha["instancia"] == caminho.stem and linha["n"] == instancia.dimensao
    geometricas = [
        "razao_aspecto",
        "fracao_fecho_convexo",
        "entropia_espacial",
        "clark_evans",
    ]
    tem_pontos = instancia.coordenadas is not None or usar_mds
    for nome in geometricas:
        assert math.isfinite(linha[nome]) == tem_pontos, nome
    assert 0 < linha["fracao_fecho_convexo"] <= 1 or not tem_pontos
    for nome in ["grau_max_agm", "fracao_folhas_agm", "cv_vizinho_mais_proximo"]:
        assert math.isfinite(linha[nome])
    assert linha["grau_max_agm"] >= 2
    assert (linha["origem_coordenadas"] == "MDS") == (
        usar_mds and instancia.coordenadas is None
    )
