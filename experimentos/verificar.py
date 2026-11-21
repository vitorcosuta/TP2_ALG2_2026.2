"""Confere se a coleta está completa e consistente antes das análises.

Compara o que está em ``resultados/`` com o que o plano manda coletar
(EXPERIMENTOS.md, seção 6) e lista o que falta, o que está duplicado e o que
foi rodado com parâmetros diferentes dos oficiais. Não roda nenhum algoritmo.

Uso (a partir da raiz do repositório, depois do ``--consolidar``)::

    python -m experimentos.verificar

Sai com código 1 se houver qualquer problema.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from experimentos import aproximados_aleatorios, bnb_aleatorias
from experimentos.algoritmos import ALGORITMOS
from experimentos.branch_and_bound import PASTA_INSTANCIAS, PASTA_RESULTADOS, ler_csv
from experimentos.instancias import (
    ALEATORIO,
    EXTRA,
    EXTRAS_REGRESSAO,
    PRINCIPAL,
    SELECIONADAS,
    nome_aleatoria,
)

LIMITE_TEMPO_OFICIAL = 1800.0
LIMITE_INFERIOR_OFICIAL = "held_karp"
MOSTRAR = 6  # problemas listados por verificação

Linhas = list[dict[str, str]]


def _instancias_tsplib() -> list[str]:
    return sorted(p.stem for p in PASTA_INSTANCIAS.glob("*.tsp"))


def _faltando_e_duplicados(
    esperado: Iterable[tuple[str, ...]], obtido: Iterable[tuple[str, ...]]
) -> list[str]:
    contagem = Counter(obtido)
    problemas = [f"falta {k}" for k in esperado if k not in contagem]
    problemas += [f"duplicado {k} ({c}×)" for k, c in contagem.items() if c > 1]
    return problemas


# ------------------------------------------------------------ verificações
def _arquivo(pasta: Path, nome: str) -> Linhas | None:
    caminho = pasta / nome
    return ler_csv(caminho) if caminho.exists() else None


def verificar_features(pasta: Path) -> list[str]:
    linhas = _arquivo(pasta, "features.csv")
    if linhas is None:
        return ["features.csv não existe (passo 1)"]
    esperado = [(i,) for i in _instancias_tsplib()]
    return _faltando_e_duplicados(esperado, [(x["instancia"],) for x in linhas])


def _verificar_tsplib(pasta: Path, nome: str) -> list[str]:
    linhas = _arquivo(pasta, nome)
    if linhas is None:
        return [f"{nome} não existe (passo 2)"]
    esperado = [(i, a) for i in _instancias_tsplib() for a in ALGORITMOS]
    obtido = [(x["instancia"], x["algoritmo"]) for x in linhas]
    if nome.startswith("subrotinas"):  # várias etapas por par: só a presença
        obtido = list(dict.fromkeys(obtido))
    return _faltando_e_duplicados(esperado, obtido)


def verificar_aleatorias_e1(pasta: Path) -> list[str]:
    nome = aproximados_aleatorios.NOME_SAIDA
    linhas = _arquivo(pasta, nome)
    if linhas is None:
        return [f"{nome} não existe (passo 2)"]
    esperado = [
        (str(n), a) for n in aproximados_aleatorios.TAMANHOS for a in ALGORITMOS
    ]
    obtido = list(dict.fromkeys((x["n"], x["algoritmo"]) for x in linhas))
    return _faltando_e_duplicados(esperado, obtido)


def _ler_bnb(pasta: Path) -> Linhas:
    return [x for caminho in sorted(pasta.glob("bnb*.csv")) for x in ler_csv(caminho)]


def _parametros_oficiais(linhas: Linhas) -> list[str]:
    return [
        f"{x['instancia']}/{x['estrategia']}: limite {x['limite_tempo_s']} s, "
        f"{x['limite_inferior']}"
        for x in linhas
        if float(x["limite_tempo_s"]) != LIMITE_TEMPO_OFICIAL
        or x["limite_inferior"] != LIMITE_INFERIOR_OFICIAL
    ]


def _chaves(linhas: Linhas, conjunto: str) -> list[tuple[str, ...]]:
    return [
        (x["instancia"], x["estrategia"]) for x in linhas if x["conjunto"] == conjunto
    ]


def verificar_bnb(pasta: Path) -> list[str]:
    linhas = _ler_bnb(pasta)
    if not linhas:
        return ["nenhum bnb*.csv (passos 3, 3b e 4)"]
    principal = [(i, e) for i in SELECIONADAS for e in ALGORITMOS]
    extras = [(i, "christofides") for i in EXTRAS_REGRESSAO]
    aleatorias = [
        (nome_aleatoria(n, s), "christofides")
        for n in bnb_aleatorias.TAMANHOS
        for s in range(bnb_aleatorias.SEMENTES)
    ]
    # sem_warm_start é opcional: não conta como falta nem como sobra.
    padrao = [x for x in linhas if x["estrategia"] != "sem_warm_start"]
    return (
        _faltando_e_duplicados(principal, _chaves(padrao, PRINCIPAL))
        + _faltando_e_duplicados(extras, _chaves(padrao, EXTRA))
        + _faltando_e_duplicados(aleatorias, _chaves(padrao, ALEATORIO))
        + _parametros_oficiais(linhas)
    )


def verificar_otimos(pasta: Path) -> list[str]:
    """O ``otimos_bnb.csv`` precisa ter sido gerado depois de juntar tudo."""
    otimos = _arquivo(pasta, "otimos_bnb.csv")
    if otimos is None:
        return ["otimos_bnb.csv não existe (passo 5: --consolidar)"]
    execucoes = Counter(
        x["instancia"] for x in _ler_bnb(pasta) if x["conjunto"] != ALEATORIO
    )
    no_arquivo = {x["instancia"]: int(x["execucoes"]) for x in otimos}
    if no_arquivo != dict(execucoes):
        return ["otimos_bnb.csv desatualizado: rode --consolidar de novo"]
    return []


def verificar_maquina_de_referencia(pasta: Path) -> list[str]:
    """Os tempos comparados (E1, E2) devem vir de um só processador."""
    fontes = {
        nome: _arquivo(pasta, nome) or []
        for nome in (
            "execucoes.csv",
            "subrotinas.csv",
            aproximados_aleatorios.NOME_SAIDA,
        )
    }
    bnb = _ler_bnb(pasta)
    fontes["bnb (pequenas, extras, aleatórias)"] = [
        x for x in bnb if x["conjunto"] != PRINCIPAL or int(x["n"]) <= 40
    ]
    processadores = {
        nome: {x.get("processador", "") for x in linhas}
        for nome, linhas in fontes.items()
    }
    todos = set().union(*processadores.values())
    if len(todos) <= 1:
        return []
    return [f"{nome}: {sorted(p)}" for nome, p in processadores.items()]


VERIFICACOES: dict[str, Callable[[Path], list[str]]] = {
    "features (passo 1)": verificar_features,
    "execucoes.csv (passo 2)": lambda p: _verificar_tsplib(p, "execucoes.csv"),
    "execucoes_sem_gc.csv (passo 2)": lambda p: _verificar_tsplib(
        p, "execucoes_sem_gc.csv"
    ),
    "subrotinas.csv (passo 2)": lambda p: _verificar_tsplib(p, "subrotinas.csv"),
    "subrotinas_sem_gc.csv (passo 2)": lambda p: _verificar_tsplib(
        p, "subrotinas_sem_gc.csv"
    ),
    "subrotinas_aleatorias_e1.csv (passo 2)": verificar_aleatorias_e1,
    "B&B: principal, extras e aleatórias (passos 3, 3b, 4)": verificar_bnb,
    "otimos_bnb.csv (passo 5)": verificar_otimos,
    "tempos comparados vêm de um só processador": verificar_maquina_de_referencia,
}


def verificar(pasta: Path = PASTA_RESULTADOS) -> dict[str, list[str]]:
    """Problemas encontrados por verificação (lista vazia = ok)."""
    return {nome: funcao(pasta) for nome, funcao in VERIFICACOES.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pasta", type=Path, default=PASTA_RESULTADOS)
    args = p.parse_args(argv)
    resultado = verificar(args.pasta)
    for nome, problemas in resultado.items():
        print(f"{'✗' if problemas else '✓'} {nome}")
        for problema in problemas[:MOSTRAR]:
            print(f"    {problema}")
        if len(problemas) > MOSTRAR:
            print(f"    … e mais {len(problemas) - MOSTRAR}")
    total = sum(map(len, resultado.values()))
    print("\nColeta completa." if not total else f"\n{total} problema(s).")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
