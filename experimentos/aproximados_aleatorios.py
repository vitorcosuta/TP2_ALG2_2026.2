"""Coleta do E1: tempo e memória dos 5 algoritmos construtivos em aleatórias.

Mede, por etapa, as 3 heurísticas (trivial, vizinho mais próximo, inserção
pelo fecho convexo) e os 2 aproximados (Twice-around-the-tree, Christofides)
em instâncias aleatórias com N de 10 a 1000, e grava
``resultados/subrotinas_aleatorias_e1.csv``. A regressão e os gráficos ficam
na camada de análise (``analises/e1.py``).

As instâncias vêm de ``experimentos.instancias.instancia_aleatoria`` (semente
0), as mesmas de qualquer máquina. Como têm coordenadas, a inserção pelo fecho
não passa pelo MDS.

Uso (a partir da raiz do repositório)::

    python -m experimentos.aproximados_aleatorios   # coleta (máquina de referência)
    python -m analises.e1                           # análise (qualquer máquina)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experimentos.algoritmos import ALGORITMOS, TIPO, Algoritmo
from experimentos.ambiente import info_ambiente
from experimentos.instancias import instancia_aleatoria
from tsp.agm import agm_prim
from tsp.christofides import vertices_grau_impar
from tsp.instancia import Instancia
from tsp.medicao import Medicao, medir

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "resultados"
NOME_SAIDA = "subrotinas_aleatorias_e1.csv"
NOME_SAIDA_BRUTOS = "tempos_brutos_e1.csv"
TAMANHOS = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
REPETICOES = 15
SEMENTE = 0


def _medir_algoritmo(
    algoritmo: Algoritmo, instancia: Instancia, repeticoes: int
) -> Medicao:
    """Mede ``algoritmo(instancia)``; isola ``instancia`` do laço que chama."""
    return medir(
        lambda m: algoritmo(instancia, m),
        repeticoes=repeticoes,
        medir_memoria=True,
    )


def _salvar_csv(caminho: Path, linhas: list[dict[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)


def coletar_dados_aleatorios(
    pasta_saida: Path = PASTA_SAIDA,
    tamanhos: list[int] = TAMANHOS,
    repeticoes: int = REPETICOES,
) -> Path:
    """Mede os 5 algoritmos em cada tamanho e salva o CSV de sub-rotinas."""
    brutos: list[dict[str,Any]] = []
    subrotinas: list[dict[str, Any]] = []
    for n in tamanhos:
        inst = instancia_aleatoria(n, SEMENTE)
        k_impares = len(vertices_grau_impar(agm_prim(inst.custos)))
        for nome_alg, func in ALGORITMOS.items():
            med = _medir_algoritmo(func, inst, repeticoes)
            for linha in med.linhas():
                subrotinas.append(
                    {
                        "algoritmo": nome_alg,
                        "tipo": TIPO[nome_alg],
                        "instancia": inst.nome,
                        "n": n,
                        "k_impares": k_impares,
                        **linha,
                        **info_ambiente(),
                    }
                )
            for etapa, lista_tempos in med.tempos_s.items():
                for rep_idx, tempo_s in enumerate(lista_tempos, start=1):
                    brutos.append(
                        {
                            "algoritmo": nome_alg,
                            "tipo": TIPO[nome_alg],
                            "instancia": inst.nome,
                            "n": n,
                            "k_impares": k_impares,
                            "etapa": etapa,
                            "repeticao": rep_idx,
                            "tempo_s": tempo_s,
                        }
                    )
        print(f"Coletado: N={n:4d} | k_impares={k_impares:3d} | reps={repeticoes}")

    caminho_csv = pasta_saida / NOME_SAIDA
    _salvar_csv(caminho_csv, subrotinas)
    _salvar_csv(pasta_saida / NOME_SAIDA_BRUTOS, brutos)
    return caminho_csv


if __name__ == "__main__":
    caminho = coletar_dados_aleatorios()
    print(f"Dados em {caminho}. Análise: python -m analises.e1")
