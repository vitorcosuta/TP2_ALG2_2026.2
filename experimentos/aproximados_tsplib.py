"""Mede os 5 algoritmos construtivos em todas as instâncias da TSPLIB.

São as 3 heurísticas de warm-start (trivial, vizinho mais próximo, inserção
pelo fecho convexo) e os 2 aproximados (Twice-around-the-tree, Christofides).
Gera as tabelas ``execucoes`` (uma linha por algoritmo × instância) e
``subrotinas`` (uma linha por algoritmo × instância × etapa), com e sem GC:

- ``resultados/execucoes.csv`` e ``resultados/execucoes_sem_gc.csv``
- ``resultados/subrotinas.csv`` e ``resultados/subrotinas_sem_gc.csv``

Uso (a partir da raiz do repositório)::

    python -m experimentos.aproximados_tsplib
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from experimentos.algoritmos import ALGORITMOS, TIPO, Algoritmo, origem_pontos
from tsp.agm import agm_prim, custo_agm
from tsp.christofides import vertices_grau_impar
from tsp.instancia import Instancia, carregar_instancia
from tsp.medicao import Medicao, medir
from tsp.tour import custo_tour

REPETICOES = 15
RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "resultados"


def _medir_algoritmo(
    algoritmo: Algoritmo,
    instancia: Instancia,
    repeticoes: int,
    desativar_gc: bool,
) -> Medicao:
    """Mede ``algoritmo(instancia)``; isola ``instancia`` do laço que chama."""
    return medir(
        lambda m: algoritmo(instancia, m),
        repeticoes=repeticoes,
        medir_memoria=True,
        desativar_gc=desativar_gc,
    )


def _custo_opt_tsplib(arquivo_tsp: Path, custos: np.ndarray) -> int | str:
    """Calcula o custo do .opt.tour na matriz métrica atual (gap preliminar)."""
    opt = arquivo_tsp.with_suffix(".opt.tour")
    if not opt.exists():
        return ""
    tokens = opt.read_text().split()
    tour = []
    for token in tokens[tokens.index("TOUR_SECTION") + 1 :]:
        if token in ("-1", "EOF"):
            break
        tour.append(int(token) - 1)
    return custo_tour(custos, tour)


def _salvar_csv(caminho: Path, linhas: list[dict[str, Any]]) -> None:
    if not linhas:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)


def rodar_tsplib(pasta_instancias: Path, desativar_gc: bool = False) -> None:
    execucoes: list[dict[str, Any]] = []
    subrotinas: list[dict[str, Any]] = []

    arquivos = sorted(pasta_instancias.glob("*.tsp"))

    for arquivo in arquivos:
        inst = carregar_instancia(arquivo)
        custos = inst.custos
        n = inst.dimensao
        nome_inst = inst.nome.removesuffix(".tsp")  # algumas têm NAME com ".tsp"

        pai = agm_prim(custos)
        c_agm = custo_agm(custos, pai)
        k_impares = len(vertices_grau_impar(pai))
        c_opt = _custo_opt_tsplib(arquivo, custos)

        for nome_alg, func in ALGORITMOS.items():
            med = _medir_algoritmo(func, inst, REPETICOES, desativar_gc)
            custo_sol, _ = med.resultado
            gap_tsplib = (
                round((custo_sol - c_opt) / c_opt, 6)
                if isinstance(c_opt, int) and c_opt > 0
                else ""
            )

            execucoes.append(
                {
                    "algoritmo": nome_alg,
                    "tipo": TIPO[nome_alg],
                    "instancia": nome_inst,
                    "n": n,
                    "k_impares": k_impares,
                    "origem_pontos": origem_pontos(inst, nome_alg),
                    "custo": custo_sol,
                    "custo_agm": c_agm,
                    "razao_agm": round(custo_sol / c_agm, 6) if c_agm > 0 else 1.0,
                    "custo_opt_tsplib": c_opt,
                    "gap_tsplib": gap_tsplib,
                    "gap_bnb": "",  # preencher depois com o ótimo do B&B
                    "fechamento_alterou": inst.fechamento_alterou,
                    "tempo_mediana_s": median(med.tempos_s["total"]),
                    "pico_memoria_bytes": med.picos_memoria_bytes["total"],
                    "gc_desativado": desativar_gc,
                    "status": "OK",
                }
            )

            for linha in med.linhas():
                subrotinas.append(
                    {
                        "algoritmo": nome_alg,
                        "tipo": TIPO[nome_alg],
                        "instancia": nome_inst,
                        "n": n,
                        "k_impares": k_impares,
                        "gc_desativado": desativar_gc,
                        **linha,
                    }
                )

        print(f"Concluído: {nome_inst} (N={n}, k={k_impares}, gc_off={desativar_gc})")

    sufixo = "_sem_gc" if desativar_gc else ""
    _salvar_csv(PASTA_SAIDA / f"execucoes{sufixo}.csv", execucoes)
    _salvar_csv(PASTA_SAIDA / f"subrotinas{sufixo}.csv", subrotinas)


if __name__ == "__main__":
    pasta = RAIZ / "TSP_Instances"
    rodar_tsplib(pasta, desativar_gc=False)
    rodar_tsplib(pasta, desativar_gc=True)
    print(f"Arquivos salvos em: {PASTA_SAIDA.resolve()}")
