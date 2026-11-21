"""Medição de tempo e memória por sub-rotina.

Uso dentro de um algoritmo::

    def algoritmo(custos, medidor: Medidor = MEDIDOR_NULO):
        with medidor.etapa("agm"):
            pai = agm_prim(custos)
        with medidor.etapa("emparelhamento"):
            ...

Uso num experimento::

    medicoes = medir(lambda m: christofides(custos, medidor=m), repeticoes=10)

Decisões:

- **Tempo e memória em execuções separadas.** O ``tracemalloc`` intercepta
  cada alocação e deixa o código Python várias vezes mais lento; medir os dois
  juntos distorceria o tempo. ``medir`` faz ``repeticoes`` execuções só de
  tempo e uma execução extra só de memória (a memória é determinística).
- **Etapas aninhadas** ("christofides/emparelhamento") e **acumuladas**: se a
  mesma etapa roda várias vezes numa execução (ex.: o limite inferior do
  B&B), somam-se os tempos, conta-se o nº de chamadas e guarda-se o maior pico.
- **Pico de memória** de uma etapa = maior memória alocada durante a etapa
  menos a memória já alocada ao entrar nela (inclui as etapas filhas).
- **Coletas do garbage collector** por etapa, e opção de desativá-lo, para a
  discussão sobre GC pedida na especificação.
- ``MEDIDOR_NULO`` não faz nada: é o padrão dos algoritmos, para que quem só
  quer o tour não pague o custo da instrumentação.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median
from typing import Any, TypeVar

R = TypeVar("R")


@dataclass
class Etapa:
    """Totais de uma etapa numa execução."""

    nome: str
    chamadas: int = 0
    tempo_s: float = 0.0
    pico_memoria_bytes: int | None = None
    coletas_gc: int = 0


@dataclass
class _Quadro:
    """Etapa aberta na pilha durante a medição de memória."""

    base: int  # memória alocada ao entrar
    maior: int  # maior memória absoluta vista até agora


@dataclass
class Medidor:
    """Registra o tempo (ou a memória) de cada etapa de uma execução."""

    medir_memoria: bool = False
    etapas: dict[str, Etapa] = field(default_factory=dict)
    _pilha_nomes: list[str] = field(default_factory=list)
    _pilha_memoria: list[_Quadro] = field(default_factory=list)

    @contextmanager
    def etapa(self, nome: str) -> Iterator[None]:
        caminho = "/".join([*self._pilha_nomes, nome])
        registro = self.etapas.setdefault(caminho, Etapa(caminho))
        self._pilha_nomes.append(nome)
        coletas_antes = _coletas_gc()
        if self.medir_memoria:
            self._abrir_memoria()
        inicio = time.perf_counter()
        try:
            yield
        finally:
            registro.tempo_s += time.perf_counter() - inicio
            if self.medir_memoria:
                pico = self._fechar_memoria()
                anterior = registro.pico_memoria_bytes or 0
                registro.pico_memoria_bytes = max(anterior, pico)
            registro.coletas_gc += _coletas_gc() - coletas_antes
            registro.chamadas += 1
            self._pilha_nomes.pop()

    def _abrir_memoria(self) -> None:
        atual, pico = tracemalloc.get_traced_memory()
        if self._pilha_memoria:  # guarda o pico do pai antes de zerá-lo
            pai = self._pilha_memoria[-1]
            pai.maior = max(pai.maior, pico)
        tracemalloc.reset_peak()
        self._pilha_memoria.append(_Quadro(base=atual, maior=atual))

    def _fechar_memoria(self) -> int:
        _, pico = tracemalloc.get_traced_memory()
        quadro = self._pilha_memoria.pop()
        maior = max(quadro.maior, pico)
        if self._pilha_memoria:
            pai = self._pilha_memoria[-1]
            pai.maior = max(pai.maior, maior)
        tracemalloc.reset_peak()
        return maior - quadro.base


class _MedidorNulo(Medidor):
    """Medidor que não registra nada (padrão dos algoritmos)."""

    @contextmanager
    def etapa(self, nome: str) -> Iterator[None]:
        yield


MEDIDOR_NULO: Medidor = _MedidorNulo()


def _coletas_gc() -> int:
    return sum(estatistica["collections"] for estatistica in gc.get_stats())


@dataclass
class Medicao:
    """Resultado de ``medir``: uma linha por etapa, pronta para virar CSV."""

    resultado: Any
    tempos_s: dict[str, list[float]]
    picos_memoria_bytes: dict[str, int]
    chamadas: dict[str, int]
    coletas_gc: dict[str, list[int]]

    def linhas(self) -> list[dict[str, Any]]:
        """Uma linha por etapa com a mediana, o mínimo e o máximo do tempo."""
        return [
            {
                "etapa": nome,
                "chamadas": self.chamadas[nome],
                "tempo_mediana_s": median(tempos),
                "tempo_min_s": min(tempos),
                "tempo_max_s": max(tempos),
                "repeticoes": len(tempos),
                "pico_memoria_bytes": self.picos_memoria_bytes.get(nome),
                "coletas_gc_mediana": median(self.coletas_gc[nome]),
            }
            for nome, tempos in self.tempos_s.items()
        ]


def _executar(
    funcao: Callable[[Medidor], R], medidor: Medidor, desativar_gc: bool
) -> R:
    gc_estava_ativo = gc.isenabled()
    if desativar_gc:
        gc.disable()
    try:
        with medidor.etapa("total"):
            return funcao(medidor)
    finally:
        if desativar_gc and gc_estava_ativo:
            gc.enable()


def medir(
    funcao: Callable[[Medidor], R],
    repeticoes: int = 10,
    medir_memoria: bool = True,
    desativar_gc: bool = False,
) -> Medicao:
    """Executa ``funcao(medidor)`` ``repeticoes`` vezes medindo o tempo e, se
    pedido, uma vez a mais medindo a memória.

    Todas as etapas ficam sob ``total``. O ``resultado`` é o da primeira
    execução de tempo.
    """
    if repeticoes < 1:
        raise ValueError("repeticoes deve ser ≥ 1")
    tempos: dict[str, list[float]] = {}
    coletas: dict[str, list[int]] = {}
    chamadas: dict[str, int] = {}
    resultado: Any = None
    for i in range(repeticoes):
        medidor = Medidor()
        saida = _executar(funcao, medidor, desativar_gc)
        if i == 0:
            resultado = saida
        for nome, etapa in medidor.etapas.items():
            tempos.setdefault(nome, []).append(etapa.tempo_s)
            coletas.setdefault(nome, []).append(etapa.coletas_gc)
            chamadas[nome] = etapa.chamadas

    picos: dict[str, int] = {}
    if medir_memoria:
        picos = picos_de_memoria(funcao, desativar_gc)
    return Medicao(resultado, tempos, picos, chamadas, coletas)


def picos_de_memoria(
    funcao: Callable[[Medidor], Any], desativar_gc: bool = False
) -> dict[str, int]:
    """Só a execução de memória de ``medir``: pico (bytes) de cada etapa.

    Útil quando a execução de tempo já foi feita e é cara demais para repetir
    (ex.: o B&B).
    """
    ja_rastreava = tracemalloc.is_tracing()
    if not ja_rastreava:
        tracemalloc.start()
    try:
        medidor = Medidor(medir_memoria=True)
        _executar(funcao, medidor, desativar_gc)
    finally:
        if not ja_rastreava:
            tracemalloc.stop()
    return {
        nome: etapa.pico_memoria_bytes or 0 for nome, etapa in medidor.etapas.items()
    }
