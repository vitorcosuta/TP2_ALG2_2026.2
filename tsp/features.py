"""Features geométricas e topológicas das instâncias (análise de sensibilidade).

Cada feature tenta capturar uma característica que pode facilitar ou atrapalhar
as heurísticas. A hipótese de cada uma está na docstring da função.

Dois grupos:

- **Da matriz de custos** (valem para todas as instâncias): estatísticas da
  AGM e das distâncias ao vizinho mais próximo. Usam ``instancia.custos``, a
  mesma matriz (com fecho métrico) que os algoritmos recebem.
- **Das coordenadas** (precisam de pontos no plano): razão de aspecto,
  proporção no fecho convexo, entropia espacial e índice de Clark–Evans.
  Instâncias sem coordenadas (gr24, fri26, gr48) ficam com NaN, a menos que se
  peça ``usar_mds=True``. Nesse caso os pontos são reconstruídos da matriz por
  MDS clássico, e a coluna ``origem_coordenadas`` registra ``"MDS"``.

Atenção na hora de interpretar:

- Pontos de ``DISPLAY_DATA_SECTION`` (bays29, bayg29, dantzig42) são só para
  desenho; os custos vêm da matriz explícita.
- Instâncias GEO têm coordenadas em latitude/longitude (graus.minutos). As
  features usam esses números como pontos no plano, o que distorce um pouco a
  geometria, mas é aceitável para regiões pequenas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, QhullError

from tsp.agm import agm_prim, graus_agm
from tsp.instancia import Coordenadas, Instancia, MatrizCustos

Vetor = npt.NDArray[np.float64]

NAN = float("nan")


# ------------------------------------------------------------ auxiliares
def _coef_variacao(valores: Vetor) -> float:
    """Desvio-padrão / média (adimensional; 0 = todos iguais)."""
    media = float(valores.mean())
    return float(valores.std() / media) if media > 0 else 0.0


def _distancias_vizinho_mais_proximo(custos: MatrizCustos) -> Vetor:
    d = custos.astype(np.float64)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ---------------------------------------------------- features da matriz
def features_agm(custos: MatrizCustos) -> dict[str, float]:
    """Estatísticas da árvore geradora mínima.

    - ``grau_max_agm``: um vértice de grau alto (hub) faz o Twice-around-the-tree
      passar por ele muitas vezes, e cada atalho pode sair caro. No Christofides,
      hubs geram muitos vértices de grau ímpar.
    - ``fracao_folhas_agm``: árvores com muitas folhas ("estrelas") se afastam
      de um caminho, que é o que o tour precisa ser.
    - ``fracao_impares_agm``: tamanho do emparelhamento do Christofides em
      relação a n. Afeta o tempo (Blossom é O(k³)) e a qualidade.
    - ``cv_arestas_agm``: arestas da AGM muito heterogêneas indicam clusters
      ligados por "pontes" longas.
    """
    n = len(custos)
    if n < 2:
        return {
            "grau_max_agm": 0, "fracao_folhas_agm": NAN,
            "fracao_impares_agm": NAN, "cv_arestas_agm": NAN, "custo_agm": 0,
        }  # fmt: skip
    pai = agm_prim(custos)
    graus = graus_agm(pai)
    filhos = np.flatnonzero(pai >= 0)
    arestas = custos[pai[filhos], filhos].astype(np.float64)
    return {
        "grau_max_agm": int(graus.max()),
        "fracao_folhas_agm": float((graus == 1).mean()),
        "fracao_impares_agm": float((graus % 2 == 1).mean()),
        "cv_arestas_agm": _coef_variacao(arestas),
        "custo_agm": int(arestas.sum()),
    }


def features_distancias(custos: MatrizCustos) -> dict[str, float]:
    """Heterogeneidade das distâncias.

    - ``cv_vizinho_mais_proximo``: pedido no enunciado. Se cada ponto tem o
      vizinho mais próximo a uma distância parecida (CV baixo), a densidade é
      uniforme. CV alto = regiões densas e regiões vazias (heterogeneidade
      local), o que costuma enganar heurísticas gulosas.
    - ``cv_distancias``: o mesmo para todas as distâncias, visão global.
    """
    n = len(custos)
    if n < 2:
        return {"cv_vizinho_mais_proximo": NAN, "cv_distancias": NAN}
    triangulo = custos[np.triu_indices(n, k=1)].astype(np.float64)
    return {
        "cv_vizinho_mais_proximo": _coef_variacao(
            _distancias_vizinho_mais_proximo(custos)
        ),
        "cv_distancias": _coef_variacao(triangulo),
    }


# ------------------------------------------------ features das coordenadas
def razao_aspecto(pontos: Coordenadas) -> float:
    """Lado menor / lado maior do bounding box, em [0, 1].

    1 = caixa quadrada; perto de 0 = pontos espalhados numa faixa estreita
    (quase uma reta, caso em que o tour ótimo é "ir e voltar").
    Usamos menor/maior (e não o contrário) para não dividir por zero em pontos
    colineares.
    """
    largura, altura = np.ptp(pontos, axis=0)
    maior = max(largura, altura)
    return float(min(largura, altura) / maior) if maior > 0 else NAN


def _fecho(pontos: Coordenadas) -> ConvexHull | None:
    try:
        return ConvexHull(pontos)
    except (QhullError, ValueError):  # < 3 pontos, todos colineares etc.
        return None


def fracao_fecho_convexo(pontos: Coordenadas) -> float:
    """Proporção dos pontos que são vértices do fecho convexo.

    O tour ótimo de uma instância euclidiana visita os vértices do fecho na
    mesma ordem em que aparecem no fecho. Quanto maior essa fração, mais da
    estrutura do ótimo já está "dada", o que favorece a heurística de inserção
    a partir do fecho.
    """
    fecho = _fecho(pontos)
    return float(len(fecho.vertices) / len(pontos)) if fecho else NAN


def entropia_espacial(pontos: Coordenadas, celulas_por_lado: int | None = None) -> float:
    """Entropia de Shannon da ocupação de uma grade, normalizada para [0, 1].

    O bounding box é dividido numa grade k × k; conta-se a fração de pontos
    em cada célula e calcula-se ``H = −Σ p log p``, dividido por ``log(k²)``.
    1 = pontos espalhados por igual; perto de 0 = concentrados em poucas
    células (clusters).

    Por padrão k = ⌈√(n/2)⌉, para ter em média ~2 pontos por célula: grades
    finas demais dão entropia alta mesmo com clusters, porque quase toda
    célula ocupada tem um ponto só.
    """
    n = len(pontos)
    k = celulas_por_lado or max(1, math.ceil(math.sqrt(n / 2)))
    if k == 1:
        return NAN
    minimo = pontos.min(axis=0)
    extensao = np.ptp(pontos, axis=0)
    extensao[extensao == 0] = 1.0
    # Pontos na borda superior caem na última célula, não numa célula k.
    celulas = np.minimum(((pontos - minimo) / extensao * k).astype(int), k - 1)
    _, contagens = np.unique(celulas[:, 0] * k + celulas[:, 1], return_counts=True)
    p = contagens / n
    return float(-(p * np.log(p)).sum() / math.log(k * k))


def indice_clark_evans(pontos: Coordenadas) -> float:
    """Média da distância ao vizinho mais próximo / valor esperado num padrão
    aleatório (Poisson) com a mesma densidade.

    O esperado é ``0,5 · √(A/n)``, com A = área do fecho convexo.
    R < 1 = agrupado (clusters); R ≈ 1 = aleatório; R > 1 = regular (grade).
    É a medida de "densidade de clustering" pedida no enunciado.
    """
    fecho = _fecho(pontos)
    n = len(pontos)
    if fecho is None or fecho.volume == 0:  # em 2D, .volume é a área
        return NAN
    diferencas = pontos[:, None, :] - pontos[None, :, :]
    d = np.hypot(diferencas[..., 0], diferencas[..., 1])
    np.fill_diagonal(d, np.inf)
    observado = float(d.min(axis=1).mean())
    esperado = 0.5 * math.sqrt(fecho.volume / n)
    return observado / esperado


def features_coordenadas(pontos: Coordenadas | None) -> dict[str, float]:
    if pontos is None:
        return {
            "razao_aspecto": NAN, "fracao_fecho_convexo": NAN,
            "entropia_espacial": NAN, "clark_evans": NAN,
        }  # fmt: skip
    return {
        "razao_aspecto": razao_aspecto(pontos),
        "fracao_fecho_convexo": fracao_fecho_convexo(pontos),
        "entropia_espacial": entropia_espacial(pontos),
        "clark_evans": indice_clark_evans(pontos),
    }


# --------------------------------------------------------------------- MDS
def mds_classico(custos: MatrizCustos) -> tuple[Coordenadas, float]:
    """Reconstrói pontos no plano a partir de uma matriz de distâncias.

    MDS clássico (Torgerson): com D² a matriz dos quadrados das distâncias e
    J = I − 11ᵀ/n, a matriz ``B = −½ J D² J`` é a matriz de produtos internos
    dos pontos centralizados. Os dois maiores autovetores de B, escalados pela
    raiz dos autovalores, dão as coordenadas 2D que melhor preservam B.

    Devolve ``(pontos, qualidade)``: qualidade = fração dos autovalores
    positivos explicada pelas duas dimensões. Perto de 1 = a matriz é quase
    euclidiana plana; bem abaixo de 1 = as coordenadas são uma aproximação
    grosseira e as features geométricas devem ser lidas com cautela.
    """
    d2 = custos.astype(np.float64) ** 2
    n = len(d2)
    j = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * j @ d2 @ j
    autovalores, autovetores = np.linalg.eigh(b)  # ordem crescente
    topo = autovalores[::-1][:2].clip(min=0)
    pontos = autovetores[:, ::-1][:, :2] * np.sqrt(topo)
    positivos = autovalores[autovalores > 0].sum()
    qualidade = float(topo.sum() / positivos) if positivos > 0 else NAN
    return pontos, qualidade


# ------------------------------------------------------------------ tudo
def extrair_features(instancia: Instancia, usar_mds: bool = False) -> dict[str, Any]:
    """Uma linha do ``features.csv`` para a instância."""
    pontos = instancia.coordenadas
    origem = instancia.origem_coordenadas
    qualidade_mds = NAN
    if pontos is None and usar_mds:
        pontos, qualidade_mds = mds_classico(instancia.custos)
        origem = "MDS"
    return {
        "instancia": instancia.nome.removesuffix(".tsp"),
        "n": instancia.dimensao,
        "tipo_peso": instancia.tipo_peso,
        "fechamento_alterou": instancia.fechamento_alterou,
        "origem_coordenadas": origem or "",
        "qualidade_mds": qualidade_mds,
        **features_coordenadas(pontos),
        **features_agm(instancia.custos),
        **features_distancias(instancia.custos),
    }