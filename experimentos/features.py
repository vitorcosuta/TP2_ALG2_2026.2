"""Gera ``resultados/features.csv``: uma linha por instância.

Uso::

    python -m experimentos.features            # sem coordenadas → NaN
    python -m experimentos.features --mds      # reconstrói pontos por MDS
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tsp.features import extrair_features
from tsp.instancia import carregar_instancia

RAIZ = Path(__file__).resolve().parent.parent
PASTA_INSTANCIAS = RAIZ / "TSP_Instances"
SAIDA_PADRAO = RAIZ / "resultados" / "features.csv"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mds", action="store_true", help="MDS onde faltam coordenadas")
    p.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    p.add_argument("--pasta", type=Path, default=PASTA_INSTANCIAS)
    args = p.parse_args(argv)

    instancias = [carregar_instancia(c) for c in sorted(args.pasta.glob("*.tsp"))]
    instancias.sort(key=lambda i: i.dimensao)
    linhas = [extrair_features(i, usar_mds=args.mds) for i in instancias]

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"{len(linhas)} instâncias → {args.saida}")


if __name__ == "__main__":
    main()
