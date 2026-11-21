"""Script de testes de Christofides e Twice-around-the-tree nas instâncias da TSPLIB.

Uso (a partir da raiz do repositório)::

    python -m experimentos.aproximados_tsplib
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from tsp.agm import agm_prim, custo_agm
from tsp.christofides import christofides, vertices_grau_impar
from tsp.instancia import carregar_instancia
from tsp.medicao import Medicao, medir
from tsp.tour import custo_tour
from tsp.twice_around_the_tree import twice_around_the_tree

REPETICOES = 15
RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "resultados"

ALGORITMOS = {
    "twice_around_the_tree": twice_around_the_tree,
    "christofides": christofides,
}


def _medir_algoritmo(
    algoritmo: Callable[..., Any],
    custos: np.ndarray,
    repeticoes: int,
    desativar_gc: bool,
) -> Medicao:
    """Mede ``algoritmo(custos)``; a função isola ``custos`` do laço que chama."""
    return medir(
        lambda m: algoritmo(custos, medidor=m),
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
            med = _medir_algoritmo(func, custos, REPETICOES, desativar_gc)
            custo_sol, _ = med.resultado
            gap_tsplib = (
                round((custo_sol - c_opt) / c_opt, 6)
                if isinstance(c_opt, int) and c_opt > 0
                else ""
            )

            execucoes.append(
                {
                    "algoritmo": nome_alg,
                    "instancia": nome_inst,
                    "n": n,
                    "k_impares": k_impares,
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
                        "instancia": nome_inst,
                        "n": n,
                        "k_impares": k_impares,
                        "gc_desativado": desativar_gc,
                        **linha,
                    }
                )

        print(f"Concluído: {nome_inst} (N={n}, k={k_impares}, gc_off={desativar_gc})")

    sufixo = "_sem_gc" if desativar_gc else ""
    _salvar_csv(PASTA_SAIDA / f"execucoes_aproximados{sufixo}.csv", execucoes)
    _salvar_csv(PASTA_SAIDA / f"subrotinas_aproximados{sufixo}.csv", subrotinas)


if __name__ == "__main__":
    pasta = RAIZ / "TSP_Instances"
    rodar_tsplib(pasta, desativar_gc=False)
    rodar_tsplib(pasta, desativar_gc=True)
    print(f"Arquivos salvos em: {PASTA_SAIDA.resolve()}")
