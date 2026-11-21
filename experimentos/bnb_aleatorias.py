"""Coleta do E1: B&B em instâncias aleatórias (conjunto ``aleatorio``).

Gera ``resultados/bnb_aleatorias.csv`` com as mesmas colunas do ``bnb.csv``;
a coluna ``conjunto`` vale ``aleatorio`` e ``semente`` identifica a instância.
Serve à regressão do B&B (tempo e nós × N) com uma curva "limpa": vários N,
várias sementes por N (o tempo varia muito entre instâncias do mesmo N).

Uso (a partir da raiz do repositório, na máquina de referência)::

    python -m experimentos.bnb_aleatorias                     # padrão abaixo
    python -m experimentos.bnb_aleatorias --tamanhos 10 20 30 --sementes 3

Instâncias: ``experimentos.instancias.instancia_aleatoria`` (pontos uniformes,
determinísticas a partir de (N, semente)). Mesmas regras de retomada e de
limite de tempo do ``branch_and_bound``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experimentos.branch_and_bound import (
    ESTRATEGIAS,
    PASTA_RESULTADOS,
    execucoes_feitas,
    executar_instancia,
    rodar_tarefas,
)
from experimentos.instancias import ALEATORIO, instancia_aleatoria, nome_aleatoria
from tsp.branch_and_bound import LimiteInferior

SAIDA_PADRAO = PASTA_RESULTADOS / "bnb_aleatorias.csv"
TAMANHOS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
SEMENTES = 5


def executar_aleatoria(
    n: int,
    semente: int,
    estrategia: str,
    limite_tempo_s: float,
    limite_inferior: LimiteInferior,
    limite_memoria_s: float,
) -> dict[str, Any]:
    """Gera a instância no próprio processo e roda warm-start + B&B."""
    linha = executar_instancia(
        instancia_aleatoria(n, semente),
        estrategia,
        limite_tempo_s,
        limite_inferior,
        limite_memoria_s,
    )
    linha.update(conjunto=ALEATORIO, semente=semente)
    return linha


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tamanhos", nargs="+", type=int, default=TAMANHOS)
    p.add_argument("--sementes", type=int, default=SEMENTES, help="por tamanho")
    p.add_argument(
        "--estrategias", nargs="+", default=["christofides"], choices=list(ESTRATEGIAS)
    )
    p.add_argument("--limite-tempo", type=float, default=1800.0)
    p.add_argument(
        "--limite-inferior",
        default="held_karp",
        choices=["dois_menores", "agm", "held_karp"],
    )
    p.add_argument("--limite-memoria", type=float, default=120.0)
    p.add_argument("--processos", type=int, default=1)
    p.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _argumentos(argv)
    feitas = execucoes_feitas(args.saida, args.limite_tempo)
    parametros = (args.limite_tempo, args.limite_inferior, args.limite_memoria)
    tarefas = [
        (n, semente, estrategia, *parametros)
        for n in sorted(args.tamanhos)
        for semente in range(args.sementes)
        for estrategia in args.estrategias
        if (
            nome_aleatoria(n, semente),
            estrategia,
            args.limite_inferior,
            args.limite_tempo,
        )
        not in feitas
    ]
    total = len(args.tamanhos) * args.sementes * len(args.estrategias)
    if total > len(tarefas):
        print(f"{total - len(tarefas)} combinações já estão em {args.saida}; pulando.")
    rodar_tarefas(executar_aleatoria, tarefas, args.saida, args.processos)


if __name__ == "__main__":
    main()
