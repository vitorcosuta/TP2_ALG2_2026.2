"""Executa o Branch-and-Bound em cada (instância × estratégia de warm-start).

Gera ``resultados/bnb.csv`` (uma linha por execução) e, ao final,
``resultados/otimos_bnb.csv`` (uma linha por instância, com o ótimo de
referência para os gaps). O formato das colunas está em ``EXPERIMENTOS.md``,
seção 2.1.

Uso (a partir da raiz do repositório)::

    python -m experimentos.branch_and_bound                      # tudo
    python -m experimentos.branch_and_bound --processos 4        # 4 em paralelo
    python -m experimentos.branch_and_bound --instancias burma14 gr17 \\
        --saida resultados/bnb_maquina2.csv                      # divisão
    python -m experimentos.branch_and_bound --consolidar         # só otimos

Decisões:

- **Retomável.** Cada linha é gravada assim que a execução termina. Ao
  reiniciar, as combinações já presentes no CSV são puladas (``--refazer``
  ignora isso). Uma queda após horas de execução não perde nada.
- **Uma execução de tempo por combinação** (30 min no máximo). A memória é
  medida numa execução **separada**, com ``tracemalloc``, só quando a de tempo
  terminou em até ``--limite-memoria`` segundos: o ``tracemalloc`` deixa o
  código 2–3× mais lento e repetir execuções de 30 min não cabe no prazo.
- **Tempo da heurística**: mediana de 10 execuções (são milissegundos). Entra
  no trade-off do E2 (heurística + B&B).
- **Paralelismo** (``--processos``): cada combinação roda num processo.
  Processos simultâneos disputam cache e memória, o que pode inflar os tempos;
  para os números finais do E1, preferir 1 processo por núcleo físico livre.
- **Divisão entre máquinas**: cada máquina usa ``--instancias`` e uma
  ``--saida`` própria (``resultados/bnb_<nome>.csv``). Ao final (ou com
  ``--consolidar``), todos os ``bnb*.csv`` da pasta da saída são juntados em
  ``otimos_bnb.csv``, na mesma pasta.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any

from experimentos.instancias import SELECIONADAS
from tsp.branch_and_bound import BranchAndBound, LimiteInferior, ResultadoBnB
from tsp.christofides import christofides
from tsp.heuristicas import insercao_fecho_convexo, trivial, vizinho_mais_proximo
from tsp.instancia import Instancia, carregar_instancia
from tsp.medicao import Medidor, medir, picos_de_memoria
from tsp.tour import Tour
from tsp.twice_around_the_tree import twice_around_the_tree

RAIZ = Path(__file__).resolve().parent.parent
PASTA_INSTANCIAS = RAIZ / "TSP_Instances"
PASTA_RESULTADOS = RAIZ / "resultados"
SAIDA_PADRAO = PASTA_RESULTADOS / "bnb.csv"
SAIDA_OTIMOS = PASTA_RESULTADOS / "otimos_bnb.csv"

REPETICOES_HEURISTICA = 10

Heuristica = Callable[[Instancia, Medidor], tuple[int, Tour]]

ESTRATEGIAS: dict[str, Heuristica] = {
    "trivial": lambda i, m: trivial(i.custos, medidor=m),
    "vizinho_mais_proximo": lambda i, m: vizinho_mais_proximo(i.custos, medidor=m),
    "insercao_fecho_convexo": lambda i, m: insercao_fecho_convexo(
        i.custos, i.coordenadas, medidor=m
    ),
    "twice_around_the_tree": lambda i, m: twice_around_the_tree(i.custos, medidor=m),
    "christofides": lambda i, m: christofides(i.custos, medidor=m),
}
# Fora do padrão: B&B sem limite superior inicial (custo_inicial = inf).
SEM_WARM_START = "sem_warm_start"

COLUNAS = [
    # identificação
    "instancia", "n", "estrategia", "limite_inferior", "limite_tempo_s",
    # warm-start
    "origem_pontos", "custo_inicial", "tempo_heuristica_s",
    # resultado do B&B
    "status", "custo", "limite_raiz",
    "nos_expandidos", "nos_podados", "profundidade_maxima", "melhorias",
    "chamadas_limite_inferior",
    # tempo e memória
    "tempo_bnb_s", "tempo_penalidades_s", "tempo_busca_s",
    "tempo_limite_inferior_s", "tempo_total_s",
    "pico_memoria_bytes", "pico_memoria_busca_bytes",
    # auditoria
    "tour", "processador", "nucleos_fisicos", "nucleos_logicos", "python", "data",
]  # fmt: skip


# ------------------------------------------------------------ hardware
def _comando(*args: str) -> str:
    try:
        saida = subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=False
        )
        return saida.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _processador_linux() -> tuple[str, int | None]:
    """Modelo e nº de núcleos físicos lidos de ``/proc/cpuinfo``."""
    try:
        texto = Path("/proc/cpuinfo").read_text()
    except OSError:
        return "", None
    modelo, nucleos = "", set()
    fisico = ""
    for linha in texto.splitlines():
        chave, _, valor = (x.strip() for x in linha.partition(":"))
        if chave == "model name" and not modelo:
            modelo = valor
        elif chave == "physical id":
            fisico = valor
        elif chave == "core id":
            nucleos.add((fisico, valor))
    return modelo, len(nucleos) or None


def _processador_macos() -> tuple[str, int | None]:
    modelo = _comando("sysctl", "-n", "machdep.cpu.brand_string").strip()
    fisicos = _comando("sysctl", "-n", "hw.physicalcpu").strip()
    return modelo, int(fisicos) if fisicos.isdigit() else None


@cache
def info_processador() -> dict[str, Any]:
    """Modelo do processador e nº de núcleos, para o relatório.

    Núcleos físicos ficam vazios quando o sistema não informa (ex.: Windows);
    os lógicos incluem hyper-threading.
    """
    sistema = platform.system()
    modelo, fisicos = "", None
    if sistema == "Linux":
        modelo, fisicos = _processador_linux()
    elif sistema == "Darwin":
        modelo, fisicos = _processador_macos()
    return {
        "processador": modelo or platform.processor() or platform.machine(),
        "nucleos_fisicos": "" if fisicos is None else fisicos,
        "nucleos_logicos": os.cpu_count() or "",
    }


# ------------------------------------------------------------ uma execução
def _warm_start(inst: Instancia, estrategia: str) -> tuple[Tour | None, float, str]:
    """Tour inicial, mediana do tempo da heurística e origem dos pontos."""
    if estrategia == SEM_WARM_START:
        return None, 0.0, ""
    heuristica = ESTRATEGIAS[estrategia]
    med = medir(
        lambda m: heuristica(inst, m),
        repeticoes=REPETICOES_HEURISTICA,
        medir_memoria=False,
    )
    _, tour = med.resultado
    origem = ""
    if estrategia == "insercao_fecho_convexo":
        origem = inst.origem_coordenadas or "MDS"
    return tour, median(med.tempos_s["total"]), origem


def _linha(
    inst: Instancia, estrategia: str, bnb: BranchAndBound, r: ResultadoBnB
) -> dict[str, Any]:
    return {
        "instancia": inst.nome.removesuffix(".tsp"),
        "n": inst.dimensao,
        "estrategia": estrategia,
        "limite_inferior": bnb.tipo_limite,
        "limite_tempo_s": bnb.limite_tempo_s,
        "status": r.status,
        "custo": "" if r.custo is None else r.custo,
        "custo_inicial": "" if r.custo_inicial == float("inf") else r.custo_inicial,
        "limite_raiz": r.limite_raiz,
        "nos_expandidos": r.nos_expandidos,
        "nos_podados": r.nos_podados,
        "profundidade_maxima": r.profundidade_maxima,
        "melhorias": r.melhorias,
        "chamadas_limite_inferior": r.chamadas_limite_inferior,
        "tempo_bnb_s": r.tempo_s,
        "tempo_penalidades_s": r.tempo_penalidades_s,
        "tempo_limite_inferior_s": r.tempo_limite_inferior_s,
        "tour": "-".join(map(str, r.tour)),
        **info_processador(),
        "python": platform.python_version(),
        "data": datetime.now().isoformat(timespec="seconds"),
    }


def executar(
    nome: str,
    estrategia: str,
    limite_tempo_s: float,
    limite_inferior: LimiteInferior,
    limite_memoria_s: float,
) -> dict[str, Any]:
    """Roda uma combinação e devolve a linha do CSV (roda em outro processo)."""
    inst = carregar_instancia(PASTA_INSTANCIAS / f"{nome}.tsp")
    tour_inicial, tempo_heuristica, origem = _warm_start(inst, estrategia)

    bnb = BranchAndBound(
        inst.custos, limite_tempo_s, limite_inferior, cronometrar_limite=True
    )
    med = medir(lambda m: bnb.resolver(tour_inicial, m), 1, medir_memoria=False)
    r: ResultadoBnB = med.resultado
    linha = _linha(inst, estrategia, bnb, r)
    linha.update(
        origem_pontos=origem,
        tempo_heuristica_s=tempo_heuristica,
        tempo_busca_s=med.tempos_s["total/busca"][0],
        tempo_total_s=tempo_heuristica + r.tempo_s,
        pico_memoria_bytes="",
        pico_memoria_busca_bytes="",
    )
    if r.status == "OK" and r.tempo_s <= limite_memoria_s:
        picos = picos_de_memoria(lambda m: bnb.resolver(tour_inicial, m))
        linha["pico_memoria_bytes"] = picos["total"]
        linha["pico_memoria_busca_bytes"] = picos["total/busca"]
    return linha


# ------------------------------------------------------------------ CSV
def _ler_csv(caminho: Path) -> list[dict[str, str]]:
    if not caminho.exists():
        return []
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def _anexar(caminho: Path, linha: dict[str, Any]) -> None:
    novo = not caminho.exists()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, COLUNAS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def _chave(linha: dict[str, Any]) -> tuple[str, str, str]:
    return linha["instancia"], linha["estrategia"], linha["limite_inferior"]


# ----------------------------------------------------------- consolidação
def _resumo_da_instancia(nome: str, linhas: list[dict[str, str]]) -> dict[str, Any]:
    custos_ok = {int(x["custo"]) for x in linhas if x["status"] == "OK"}
    if len(custos_ok) > 1:
        print(f"AVISO: {nome} tem ótimos diferentes: {custos_ok}", file=sys.stderr)
    custos = [int(x["custo"]) for x in linhas if x["custo"]]
    return {
        "instancia": nome,
        "n": linhas[0]["n"],
        "otimo_provado": bool(custos_ok),
        "melhor_custo": min(custos) if custos else "",
        "maior_limite_raiz": max(int(x["limite_raiz"]) for x in linhas),
        "execucoes": len(linhas),
        "execucoes_ok": sum(x["status"] == "OK" for x in linhas),
    }


def consolidar(arquivos: Iterable[Path], saida: Path = SAIDA_OTIMOS) -> list[dict]:
    """Uma linha por instância com o ótimo de referência para os gaps.

    ``otimo_provado`` = alguma estratégia terminou (status OK); nesse caso
    ``melhor_custo`` é o ótimo. Senão, é o melhor tour achado (cota superior),
    e ``maior_limite_raiz`` é a melhor cota inferior: o ótimo está entre os
    dois.
    """
    por_instancia: dict[str, list[dict[str, str]]] = {}
    for caminho in arquivos:
        for linha in _ler_csv(caminho):
            por_instancia.setdefault(linha["instancia"], []).append(linha)

    grupos = sorted(por_instancia.items(), key=lambda x: int(x[1][0]["n"]))
    otimos = [_resumo_da_instancia(nome, linhas) for nome, linhas in grupos]
    if otimos:
        saida.parent.mkdir(parents=True, exist_ok=True)
        with saida.open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, list(otimos[0]))
            escritor.writeheader()
            escritor.writerows(otimos)
    return otimos


# ------------------------------------------------------------------- main
def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--instancias", nargs="+", default=SELECIONADAS)
    p.add_argument(
        "--estrategias",
        nargs="+",
        default=list(ESTRATEGIAS),
        choices=[*ESTRATEGIAS, SEM_WARM_START],
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
    p.add_argument("--refazer", action="store_true")
    p.add_argument("--consolidar", action="store_true")
    return p.parse_args(argv)


def _pendentes(args: argparse.Namespace) -> list[tuple[str, str]]:
    feitas = set() if args.refazer else {_chave(x) for x in _ler_csv(args.saida)}
    tamanhos = {
        nome: carregar_instancia(PASTA_INSTANCIAS / f"{nome}.tsp").dimensao
        for nome in args.instancias
    }
    combinacoes = [
        (nome, estrategia)
        for nome in sorted(args.instancias, key=tamanhos.__getitem__)
        for estrategia in args.estrategias
        if (nome, estrategia, args.limite_inferior) not in feitas
    ]
    puladas = len(args.instancias) * len(args.estrategias) - len(combinacoes)
    if puladas:
        print(f"{puladas} combinações já estão em {args.saida}; pulando.")
    return combinacoes


def _relatar(linha: dict[str, Any], feitas: int, total: int) -> None:
    print(
        f"[{feitas}/{total}] {linha['instancia']:>10} {linha['estrategia']:<22} "
        f"{linha['status']}  custo={linha['custo']}  "
        f"nós={linha['nos_expandidos']}  {linha['tempo_bnb_s']:.1f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = _argumentos(argv)
    if not args.consolidar:
        combinacoes = _pendentes(args)
        parametros = (args.limite_tempo, args.limite_inferior, args.limite_memoria)
        with ProcessPoolExecutor(max_workers=args.processos) as executor:
            futuros = [
                executor.submit(executar, nome, estrategia, *parametros)
                for nome, estrategia in combinacoes
            ]
            for feitas, futuro in enumerate(as_completed(futuros), 1):
                linha = futuro.result()
                _anexar(args.saida, linha)
                _relatar(linha, feitas, len(combinacoes))
    # Junta todas as saídas da pasta (uma por máquina) num só arquivo de ótimos.
    pasta = args.saida.parent
    saida_otimos = pasta / SAIDA_OTIMOS.name
    otimos = consolidar(sorted(pasta.glob("bnb*.csv")), saida_otimos)
    print(f"{len(otimos)} instâncias → {saida_otimos}")


if __name__ == "__main__":
    main()
