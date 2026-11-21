"""Executa o Branch-and-Bound em cada (instância × estratégia de warm-start).

Gera ``resultados/bnb.csv`` (uma linha por execução). Com ``--consolidar``,
junta todos os ``bnb*.csv`` da pasta em ``otimos_bnb.csv`` (uma linha por
instância, com o ótimo de referência para os gaps). O formato das colunas está
em ``EXPERIMENTOS.md``, seção 2.1.

Uso (a partir da raiz do repositório)::

    python -m experimentos.branch_and_bound                      # tudo
    python -m experimentos.branch_and_bound --processos 4        # 4 em paralelo
    python -m experimentos.branch_and_bound --instancias burma14 gr17 \\
        --saida resultados/bnb_maquina2.csv                      # divisão
    python -m experimentos.branch_and_bound --consolidar         # só otimos
    python -m experimentos.branch_and_bound --instancias st70 --limite-tempo 5 \\
        --saida /tmp/teste.csv                                   # teste rápido

Decisões:

- **Retomável.** Cada linha é gravada assim que a execução termina. Ao
  reiniciar, as combinações já presentes no CSV (mesma instância, estratégia,
  limite inferior e **limite de tempo**) são puladas. Uma queda após horas de
  execução não perde nada. Para refazer, use outra ``--saida`` ou apague o
  arquivo.
- **Um arquivo, um limite de tempo.** Se a ``--saida`` já tem execuções com
  outro limite de tempo (ex.: um teste rápido com ``--limite-tempo 5``), o
  script para com erro em vez de misturar os dois. Testes rápidos devem usar
  uma ``--saida`` fora de ``resultados/``.
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
  ``--saida`` própria (``resultados/bnb_<nome>.csv``).
- **Consolidação só sob demanda** (``--consolidar``), depois de juntar no git
  os ``bnb_*.csv`` de todas as máquinas. Se rodasse ao fim de cada execução,
  cada máquina geraria um ``otimos_bnb.csv`` parcial diferente, e os commits
  entrariam em conflito ou sobrescreveriam uns aos outros.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from experimentos.algoritmos import ALGORITMOS, origem_pontos
from experimentos.ambiente import COLUNAS_AMBIENTE, info_ambiente
from experimentos.instancias import ALEATORIO, SELECIONADAS, conjunto_da_instancia
from tsp.branch_and_bound import BranchAndBound, LimiteInferior, ResultadoBnB
from tsp.instancia import Instancia, carregar_instancia
from tsp.medicao import medir, picos_de_memoria
from tsp.tour import Tour

RAIZ = Path(__file__).resolve().parent.parent
PASTA_INSTANCIAS = RAIZ / "TSP_Instances"
PASTA_RESULTADOS = RAIZ / "resultados"
SAIDA_PADRAO = PASTA_RESULTADOS / "bnb.csv"
SAIDA_OTIMOS = PASTA_RESULTADOS / "otimos_bnb.csv"

REPETICOES_HEURISTICA = 10

ESTRATEGIAS = ALGORITMOS
# Fora do padrão: B&B sem limite superior inicial (custo_inicial = inf).
SEM_WARM_START = "sem_warm_start"

COLUNAS = [
    # identificação
    "conjunto", "instancia", "semente", "n",
    "estrategia", "limite_inferior", "limite_tempo_s",
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
    "tour", *COLUNAS_AMBIENTE, "data",
]  # fmt: skip


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
    return tour, median(med.tempos_s["total"]), origem_pontos(inst, estrategia)


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
        **info_ambiente(),
        "data": datetime.now().isoformat(timespec="seconds"),
    }


def executar(
    nome: str,
    estrategia: str,
    limite_tempo_s: float,
    limite_inferior: LimiteInferior,
    limite_memoria_s: float,
) -> dict[str, Any]:
    """Roda uma combinação de instância da TSPLIB (roda em outro processo)."""
    inst = carregar_instancia(PASTA_INSTANCIAS / f"{nome}.tsp")
    linha = executar_instancia(
        inst, estrategia, limite_tempo_s, limite_inferior, limite_memoria_s
    )
    linha.update(conjunto=conjunto_da_instancia(nome), semente="")
    return linha


def executar_instancia(
    inst: Instancia,
    estrategia: str,
    limite_tempo_s: float,
    limite_inferior: LimiteInferior,
    limite_memoria_s: float,
) -> dict[str, Any]:
    """Warm-start + B&B (+ memória, se terminou rápido) numa instância pronta.

    Devolve a linha do CSV sem ``conjunto`` e ``semente``, que quem chama
    preenche.
    """
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
def ler_csv(caminho: Path) -> list[dict[str, str]]:
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


Chave = tuple[str, str, str, float]


def chave_execucao(linha: dict[str, Any]) -> Chave:
    """Identifica uma execução: instância, estratégia, limite inferior e tempo."""
    return (
        linha["instancia"],
        linha["estrategia"],
        linha["limite_inferior"],
        float(linha["limite_tempo_s"]),
    )


class LimiteDeTempoMisturado(SystemExit):
    """A saída já tem execuções com outro limite de tempo."""


def execucoes_feitas(saida: Path, limite_tempo_s: float) -> set[Chave]:
    """Chaves já gravadas em ``saida``; recusa misturar limites de tempo.

    Um teste rápido (``--limite-tempo 5``) gravado no arquivo oficial seria
    pulado pela execução real, que ficaria com o resultado de 5 s. Por isso
    um arquivo só aceita um limite de tempo.
    """
    feitas = {chave_execucao(x) for x in ler_csv(saida)}
    outros = sorted({c[3] for c in feitas} - {float(limite_tempo_s)})
    if outros:
        raise LimiteDeTempoMisturado(
            f"{saida} já tem execuções com limite de tempo {outros} s, diferente "
            f"de {limite_tempo_s} s. Use outra --saida (testes rápidos devem "
            "gravar fora de resultados/) ou apague o arquivo."
        )
    return feitas


# ----------------------------------------------------------- consolidação
def _resumo_da_instancia(nome: str, linhas: list[dict[str, str]]) -> dict[str, Any]:
    custos_ok = {int(x["custo"]) for x in linhas if x["status"] == "OK"}
    if len(custos_ok) > 1:
        print(f"AVISO: {nome} tem ótimos diferentes: {custos_ok}", file=sys.stderr)
    custos = [int(x["custo"]) for x in linhas if x["custo"]]
    return {
        "conjunto": linhas[0].get("conjunto", ""),
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
        for linha in ler_csv(caminho):
            if linha.get("conjunto") == ALEATORIO:  # não precisam de ótimo
                continue
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
    p.add_argument(
        "--consolidar",
        action="store_true",
        help="só gera otimos_bnb.csv a partir dos bnb*.csv da pasta da saída",
    )
    return p.parse_args(argv)


def _pendentes(args: argparse.Namespace) -> list[tuple[str, str]]:
    feitas = execucoes_feitas(args.saida, args.limite_tempo)
    tamanhos = {
        nome: carregar_instancia(PASTA_INSTANCIAS / f"{nome}.tsp").dimensao
        for nome in args.instancias
    }
    combinacoes = [
        (nome, estrategia)
        for nome in sorted(args.instancias, key=tamanhos.__getitem__)
        for estrategia in args.estrategias
        if (nome, estrategia, args.limite_inferior, args.limite_tempo) not in feitas
    ]
    puladas = len(args.instancias) * len(args.estrategias) - len(combinacoes)
    if puladas:
        print(f"{puladas} combinações já estão em {args.saida}; pulando.")
    return combinacoes


def _relatar(linha: dict[str, Any], feitas: int, total: int) -> None:
    print(
        f"[{feitas}/{total}] {linha['instancia']:>14} {linha['estrategia']:<22} "
        f"{linha['status']}  custo={linha['custo']}  "
        f"nós={linha['nos_expandidos']}  {linha['tempo_bnb_s']:.1f}s",
        flush=True,
    )


def rodar_tarefas(
    funcao: Callable[..., dict[str, Any]],
    tarefas: list[tuple[Any, ...]],
    saida: Path,
    processos: int,
) -> None:
    """Executa ``funcao(*tarefa)`` em processos e grava cada linha ao terminar."""
    with ProcessPoolExecutor(max_workers=processos) as executor:
        futuros = [executor.submit(funcao, *tarefa) for tarefa in tarefas]
        for feitas, futuro in enumerate(as_completed(futuros), 1):
            linha = futuro.result()
            _anexar(saida, linha)
            _relatar(linha, feitas, len(tarefas))


def main(argv: list[str] | None = None) -> None:
    args = _argumentos(argv)
    if args.consolidar:
        # Junta as saídas de todas as máquinas num só arquivo de ótimos.
        pasta = args.saida.parent
        saida_otimos = pasta / SAIDA_OTIMOS.name
        otimos = consolidar(sorted(pasta.glob("bnb*.csv")), saida_otimos)
        print(f"{len(otimos)} instâncias → {saida_otimos}")
        return
    parametros = (args.limite_tempo, args.limite_inferior, args.limite_memoria)
    tarefas = [(nome, est, *parametros) for nome, est in _pendentes(args)]
    rodar_tarefas(executar, tarefas, args.saida, args.processos)


if __name__ == "__main__":
    main()
