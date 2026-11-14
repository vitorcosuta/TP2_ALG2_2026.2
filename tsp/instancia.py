"""Leitura de instâncias TSPLIB e conversão para uma matriz de custos métrica.

O leitor devolve sempre uma matriz de custos inteiros n×n, venham os pesos de
uma ``EDGE_WEIGHT_SECTION`` ou de coordenadas (``NODE_COORD_SECTION``), seguindo
as fórmulas da especificação da TSPLIB.

Como várias instâncias violam a desigualdade triangular, o leitor aplica por
padrão o fecho métrico: sempre que ``c(u, w) > c(u, v) + c(v, w)``, o custo de
``(u, w)`` passa a ser ``c(u, v) + c(v, w)``, repetindo até não haver violação.
O resultado é a matriz de menores caminhos (Floyd–Warshall).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

MatrizCustos = npt.NDArray[np.int64]
Coordenadas = npt.NDArray[np.float64]

_CAMPO = re.compile(r"^([A-Z_]+)\s*:\s*(.*)$")
_PI_TSPLIB = 3.141592
_RAIO_TERRA = 6378.388


@dataclass(frozen=True)
class Instancia:
    """Instância do TSP pronta para os algoritmos.

    ``custos`` é a matriz usada pelos algoritmos (métrica, se ``metrica=True``
    na leitura); ``custos_originais`` guarda os pesos exatamente como a TSPLIB
    os define.

    ``coordenadas`` vem da ``NODE_COORD_SECTION`` ou, na falta dela, da
    ``DISPLAY_DATA_SECTION``; é ``None`` quando o arquivo não traz pontos.
    ``origem_coordenadas`` diz de qual seção os pontos vieram. Pontos de
    ``DISPLAY_DATA_SECTION`` servem só para as features geométricas: os custos
    dessas instâncias vêm dos pesos explícitos e não precisam coincidir com a
    distância entre os pontos.
    """

    nome: str
    dimensao: int
    tipo_peso: str
    custos: MatrizCustos
    custos_originais: MatrizCustos
    coordenadas: Coordenadas | None
    origem_coordenadas: str | None

    @property
    def fechamento_alterou(self) -> bool:
        """Indica se o fecho métrico mudou algum custo."""
        return bool((self.custos != self.custos_originais).any())


def _ler_arquivo(caminho: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    especificacao: dict[str, str] = {}
    secoes: dict[str, list[str]] = {}
    atual: list[str] | None = None
    for linha in caminho.read_text().splitlines():
        texto = linha.strip()
        if not texto or texto == "EOF":
            continue
        primeiro = texto.split()[0]
        if primeiro.endswith("_SECTION"):
            atual = secoes.setdefault(primeiro, [])
        elif campo := _CAMPO.match(texto):
            especificacao[campo.group(1)] = campo.group(2).strip()
            atual = None
        elif atual is not None:
            atual.extend(texto.split())
    return especificacao, secoes


def _matriz_explicita(valores: list[str], n: int, formato: str) -> MatrizCustos:
    pesos: Iterator[int] = (int(float(v)) for v in valores)
    custos = np.zeros((n, n), dtype=np.int64)
    if formato == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                custos[i, j] = next(pesos)
        return custos
    pares: dict[str, Callable[[int], range]] = {
        "UPPER_ROW": lambda i: range(i + 1, n),
        "UPPER_DIAG_ROW": lambda i: range(i, n),
        "LOWER_ROW": lambda i: range(i),
        "LOWER_DIAG_ROW": lambda i: range(i + 1),
    }
    if formato not in pares:
        raise ValueError(f"EDGE_WEIGHT_FORMAT não suportado: {formato}")
    for i in range(n):
        for j in pares[formato](i):
            custos[i, j] = custos[j, i] = next(pesos)
    return custos


def _nint(x: float) -> int:
    return int(x + 0.5)


def _dist_euc_2d(a: Coordenadas, b: Coordenadas) -> int:
    return _nint(math.hypot(a[0] - b[0], a[1] - b[1]))


def _dist_att(a: Coordenadas, b: Coordenadas) -> int:
    r = math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10.0)
    t = _nint(r)
    return t + 1 if t < r else t


def _radianos_geo(x: float) -> float:
    graus = int(x)
    return _PI_TSPLIB * (graus + 5.0 * (x - graus) / 3.0) / 180.0


def _dist_geo(a: Coordenadas, b: Coordenadas) -> int:
    lat_a, lon_a = _radianos_geo(a[0]), _radianos_geo(a[1])
    lat_b, lon_b = _radianos_geo(b[0]), _radianos_geo(b[1])
    q1 = math.cos(lon_a - lon_b)
    q2 = math.cos(lat_a - lat_b)
    q3 = math.cos(lat_a + lat_b)
    return int(_RAIO_TERRA * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)


_DISTANCIAS: dict[str, Callable[[Coordenadas, Coordenadas], int]] = {
    "EUC_2D": _dist_euc_2d,
    "ATT": _dist_att,
    "GEO": _dist_geo,
}


def _ler_coordenadas(valores: list[str], n: int) -> Coordenadas:
    tabela = np.array(valores, dtype=np.float64).reshape(n, 3)
    return tabela[:, 1:]


def _matriz_por_coordenadas(pontos: Coordenadas, tipo: str) -> MatrizCustos:
    if tipo not in _DISTANCIAS:
        raise ValueError(f"EDGE_WEIGHT_TYPE não suportado: {tipo}")
    distancia = _DISTANCIAS[tipo]
    n = len(pontos)
    custos = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            custos[i, j] = custos[j, i] = distancia(pontos[i], pontos[j])
    return custos


def fechamento_metrico(custos: MatrizCustos) -> MatrizCustos:
    """Devolve a matriz de menores caminhos (Floyd–Warshall), em O(n³).

    Para todo vértice intermediário ``v``, troca ``c(u, w)`` por
    ``c(u, v) + c(v, w)`` quando a soma é menor. Ao final, a desigualdade
    triangular vale para todos os trios. A matriz de entrada não é alterada.
    """
    resultado = custos.copy()
    for v in range(len(resultado)):
        np.minimum(
            resultado, resultado[:, v, None] + resultado[None, v, :], out=resultado
        )
    return resultado


def carregar_instancia(caminho: str | Path, metrica: bool = True) -> Instancia:
    """Lê um arquivo ``.tsp`` e devolve a instância com sua matriz de custos."""
    caminho = Path(caminho)
    especificacao, secoes = _ler_arquivo(caminho)
    n = int(especificacao["DIMENSION"])
    tipo = especificacao["EDGE_WEIGHT_TYPE"]

    origem = next(
        (s for s in ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION") if s in secoes),
        None,
    )
    coordenadas = _ler_coordenadas(secoes[origem], n) if origem else None

    if tipo == "EXPLICIT":
        originais = _matriz_explicita(
            secoes["EDGE_WEIGHT_SECTION"], n, especificacao["EDGE_WEIGHT_FORMAT"]
        )
    elif coordenadas is not None and origem == "NODE_COORD_SECTION":
        originais = _matriz_por_coordenadas(coordenadas, tipo)
    else:
        raise ValueError(f"{caminho.name}: sem pesos explícitos nem coordenadas")

    custos = fechamento_metrico(originais) if metrica else originais.copy()
    return Instancia(
        nome=especificacao.get("NAME", caminho.stem),
        dimensao=n,
        tipo_peso=tipo,
        custos=custos,
        custos_originais=originais,
        coordenadas=coordenadas,
        origem_coordenadas=origem,
    )
