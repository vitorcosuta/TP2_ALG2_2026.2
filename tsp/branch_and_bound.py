"""Branch-and-Bound exato para o TSP métrico, com busca em profundidade.

Árvore de busca: a raiz é o caminho ``[0]`` (o vértice 0 é fixado como início,
o que elimina as n rotações de um mesmo ciclo). Cada nó é um caminho parcial
``0 → … → atual``; seus filhos estendem o caminho com um vértice não visitado.
As folhas (profundidade n) fecham o ciclo voltando a 0.

Poda: um filho é descartado quando ``limite_inferior ≥ melhor custo``. Como os
custos são inteiros, "≥" é correto: o filho não pode gerar ciclo
estritamente melhor que o já conhecido.

Limites inferiores (custo do caminho parcial + estimativa do resto, em que o
resto é um caminho hamiltoniano de ``atual`` até 0 passando por todos os não
visitados ``U``):

- ``"dois_menores"``: cada vértice de U tem duas arestas no resto, e ``atual``
  e 0 têm uma cada. Somando as menores arestas possíveis de cada vértice e
  dividindo por 2 (cada aresta é contada pelas duas pontas), obtém-se
  ``⌈(Σ_{u∈U} (m1[u] + m2[u]) + m1[atual] + m1[0]) / 2⌉``. O(1) por nó (a soma
  é mantida incrementalmente), mas fraco.
- ``"agm"``: as arestas do resto dentro de U formam um caminho que gera U, logo
  custam ≥ AGM(U); somam-se a menor aresta de ``atual`` a U e a menor de U a 0.
  O(|U|²) por nó.
- ``"held_karp"`` (padrão): o mesmo limite de ``"agm"``, mas sobre custos
  penalizados ``c'(u, v) = c(u, v) + π[u] + π[v]``. Todo caminho que entra e sai
  de cada vértice de U tem custo' = custo + 2·Σ_{U} π + π[atual] + π[0], então
  basta descontar essa constante. As penalidades π são calculadas uma vez, na
  raiz, por otimização por subgradiente do limite de 1-árvore (Held e Karp,
  1970). O passo não depende do limite superior, para que o warm-start não
  interfira na qualidade do limite inferior.

``"agm"`` e ``"held_karp"`` usam ``"dois_menores"`` como pré-filtro: se o
limite barato já poda, o caro não é calculado.

Tempo limite: verificado a cada 4096 nós (consultar o relógio em todo nó
custaria mais que o próprio teste). Ao estourar, a busca para e o resultado
traz status ``"NA"`` **com os contadores e o melhor tour encontrado até ali**.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from tsp.instancia import MatrizCustos
from tsp.medicao import MEDIDOR_NULO, Medidor
from tsp.tour import Tour, custo_tour, eh_tour_valido

LimiteInferior = Literal["dois_menores", "agm", "held_karp"]
Vetor = npt.NDArray[np.float64]

_VERIFICAR_TEMPO_A_CADA = 4096  # potência de 2: o teste usa máscara de bits
_FOLGA_PONTO_FLUTUANTE = 1e-6


class _TempoEsgotado(Exception):
    pass


@dataclass(frozen=True)
class ResultadoBnB:
    """Resultado de uma execução do Branch-and-Bound."""

    custo: int | None  # melhor tour achado (ótimo se OK); None se nenhum
    tour: Tour
    status: Literal["OK", "NA"]  # NA = tempo limite esgotado
    custo_inicial: int | float  # limite superior do warm-start (inf se nenhum)
    limite_raiz: int  # limite inferior na raiz da árvore
    nos_expandidos: int  # nós da árvore visitados (inclui folhas)
    nos_podados: int  # filhos descartados pelo limite inferior
    profundidade_maxima: int
    melhorias: int  # vezes em que a busca achou tour melhor que o incumbente
    tempo_s: float  # penalidades + busca
    tempo_penalidades_s: float  # subgradiente de Held-Karp (0 nos outros)
    tempo_limite_inferior_s: float | None  # só se cronometrar_limite=True
    chamadas_limite_inferior: int

    @property
    def otimo_provado(self) -> bool:
        return self.status == "OK"


# --------------------------------------------------------------- Held-Karp
def _um_arvore(custos: Vetor) -> tuple[float, npt.NDArray[np.int64]]:
    """1-árvore mínima: AGM em {1..n−1} mais as duas menores arestas de 0.

    Devolve o custo e o grau de cada vértice. ``custos`` deve ter a diagonal
    infinita.
    """
    n = len(custos)
    graus = np.zeros(n, dtype=np.int64)
    distancia = custos[1, 1:].copy()
    pai = np.zeros(n - 1, dtype=np.int64)
    na_arvore = np.zeros(n - 1, dtype=bool)
    na_arvore[0] = True
    distancia[0] = np.inf
    total = 0.0
    for _ in range(n - 2):
        j = int(np.argmin(distancia))
        total += float(distancia[j])
        na_arvore[j] = True
        graus[j + 1] += 1
        graus[pai[j] + 1] += 1
        linha = custos[j + 1, 1:]
        melhora = ~na_arvore & (linha < distancia)
        distancia[melhora] = linha[melhora]
        pai[melhora] = j
        distancia[na_arvore] = np.inf
    a, b = np.argsort(custos[0, 1:])[:2] + 1
    graus[0] = 2
    graus[a] += 1
    graus[b] += 1
    return total + float(custos[0, a] + custos[0, b]), graus


def penalidades_held_karp(
    custos: MatrizCustos,
    iteracoes: int = 300,
    passo_inicial: float = 2.0,
    decaimento: float = 0.99,
) -> tuple[Vetor, float]:
    """Penalidades π que maximizam o limite de 1-árvore (subgradiente).

    Devolve ``(π, limite)``, em que ``limite`` ≤ custo do ciclo ótimo. O passo
    é ``λ · (1% do melhor limite + 1) / ‖g‖²``, com λ decaindo geometricamente;
    ``g[v] = grau(v) − 2`` na 1-árvore. Se a 1-árvore virar um ciclo, ele é
    ótimo e a otimização para.
    """
    n = len(custos)
    if n < 3:
        return np.zeros(n), float(custo_tour(custos, list(range(n))))
    base = custos.astype(np.float64)
    np.fill_diagonal(base, np.inf)
    pi = np.zeros(n)
    melhor_pi, melhor_limite = pi.copy(), -math.inf
    passo = passo_inicial
    for _ in range(iteracoes):
        custo_arvore, graus = _um_arvore(base + pi[:, None] + pi[None, :])
        limite = custo_arvore - 2 * float(pi.sum())
        if limite > melhor_limite:
            melhor_limite, melhor_pi = limite, pi.copy()
        g = graus - 2
        norma = int(g @ g)
        if norma == 0:
            break
        pi = pi + passo * (0.01 * melhor_limite + 1) / norma * g
        passo *= decaimento
    return melhor_pi, melhor_limite


# ------------------------------------------------------------ B&B
def _ordem_dos_filhos(c: list[list[int]], ordenar: bool) -> list[list[int]]:
    """Para cada vértice, os demais vértices (por custo crescente, se pedido)."""
    n = len(c)
    ordem = [[w for w in range(n) if w != v] for v in range(n)]
    if ordenar:
        for v, lista in enumerate(ordem):
            lista.sort(key=c[v].__getitem__)
    return ordem


def _menores_arestas(c: list[list[int]]) -> tuple[list[int], list[int]]:
    """Menor aresta de cada vértice e soma das duas menores (0 se faltarem)."""
    m1: list[int] = []
    m12: list[int] = []
    for v, linha in enumerate(c):
        menores = sorted(x for w, x in enumerate(linha) if w != v) + [0, 0]
        m1.append(menores[0])
        m12.append(menores[0] + menores[1])
    return m1, m12


class BranchAndBound:
    """Branch-and-Bound em profundidade com warm-start opcional.

    ``ordenar_filhos`` visita primeiro os vizinhos mais baratos de cada nó, o
    que tende a achar tours bons cedo e a podar mais.
    ``cronometrar_limite`` mede o tempo gasto no cálculo do limite inferior
    (custa ~0,1 µs por chamada; desligado por padrão).
    """

    def __init__(
        self,
        custos: MatrizCustos,
        limite_tempo_s: float = 1800.0,
        limite_inferior: LimiteInferior = "held_karp",
        ordenar_filhos: bool = True,
        cronometrar_limite: bool = False,
    ) -> None:
        if limite_inferior not in ("dois_menores", "agm", "held_karp"):
            raise ValueError(f"limite inferior desconhecido: {limite_inferior}")
        self.custos = custos
        self.c: list[list[int]] = custos.tolist()  # acesso escalar rápido
        self.n = len(custos)
        self.limite_tempo_s = limite_tempo_s
        self.tipo_limite = limite_inferior
        self.cronometrar_limite = cronometrar_limite

        self.ordem = _ordem_dos_filhos(self.c, ordenar_filhos)
        self.m1, self.m12 = _menores_arestas(self.c)

        # Preenchidos em resolver(): custos penalizados e π.
        self.pi: Vetor = np.zeros(self.n)
        self.penalizados: Vetor = custos.astype(np.float64)

    # ------------------------------------------------------------ interface
    def resolver(
        self, tour_inicial: Sequence[int] | None = None, medidor: Medidor = MEDIDOR_NULO
    ) -> ResultadoBnB:
        """Busca o ciclo ótimo; ``tour_inicial`` (warm-start) dá o limite superior."""
        self._reiniciar(tour_inicial)
        inicio = time.perf_counter()
        self._prazo = inicio + self.limite_tempo_s
        with medidor.etapa("penalidades"):
            self._preparar_penalidades()
        tempo_penalidades = time.perf_counter() - inicio
        status: Literal["OK", "NA"] = "OK"
        with medidor.etapa("busca"):
            try:
                self._buscar()
            except _TempoEsgotado:
                status = "NA"
        return ResultadoBnB(
            custo=int(self.melhor_custo) if self.melhor_tour else None,
            tour=list(self.melhor_tour),
            status=status,
            custo_inicial=self.custo_inicial,
            limite_raiz=self._limite_raiz(),
            nos_expandidos=self.nos_expandidos,
            nos_podados=self.nos_podados,
            profundidade_maxima=self.profundidade_maxima,
            melhorias=self.melhorias,
            tempo_s=time.perf_counter() - inicio,
            tempo_penalidades_s=tempo_penalidades,
            tempo_limite_inferior_s=(
                self.tempo_limite if self.cronometrar_limite else None
            ),
            chamadas_limite_inferior=self.chamadas_limite,
        )

    # -------------------------------------------------------------- estado
    def _reiniciar(self, tour_inicial: Sequence[int] | None) -> None:
        self.melhor_custo: int | float = math.inf
        self.melhor_tour: Tour = []
        if tour_inicial is not None:
            if not eh_tour_valido(tour_inicial, self.n):
                raise ValueError("tour inicial não visita cada vértice uma vez")
            self.melhor_custo = custo_tour(self.custos, tour_inicial)
            self.melhor_tour = _rotacionar_para_zero(tour_inicial)
        self.custo_inicial = self.melhor_custo
        self.nos_expandidos = 0
        self.nos_podados = 0
        self.profundidade_maxima = 0
        self.melhorias = 0
        self.tempo_limite = 0.0
        self.chamadas_limite = 0
        self.visitado = [False] * self.n
        self.visitado[0] = True
        self.caminho = [0]
        self.soma_m12_livres = sum(self.m12[1:])

    def _preparar_penalidades(self) -> None:
        self.pi = np.zeros(self.n)
        if self.tipo_limite == "held_karp":
            self.pi, _ = penalidades_held_karp(self.custos)
        self.penalizados = self.custos + self.pi[:, None] + self.pi[None, :]

    def _limite_raiz(self) -> int:
        if self.n < 2:
            return 0
        visitado = self.visitado
        self.visitado = [v == 0 for v in range(self.n)]
        soma = self.soma_m12_livres
        self.soma_m12_livres = sum(self.m12[1:])
        try:
            # Na raiz o "resto" vai de 0 até 0 passando por todos os outros.
            return self._calcular_limite(0, 0, ignorar_pre_filtro=True)
        finally:
            self.visitado, self.soma_m12_livres = visitado, soma

    def _buscar(self) -> None:
        if self.n == 1:
            self.nos_expandidos = self.profundidade_maxima = 1
            self.melhor_custo, self.melhor_tour = 0, [0]
            return
        self._dfs(0, 0, 1)

    # --------------------------------------------------------------- busca
    def _dfs(self, atual: int, custo: int, profundidade: int) -> None:
        self.nos_expandidos += 1
        if (
            self.nos_expandidos & (_VERIFICAR_TEMPO_A_CADA - 1) == 0
            and time.perf_counter() > self._prazo
        ):
            raise _TempoEsgotado
        self.profundidade_maxima = max(self.profundidade_maxima, profundidade)

        if profundidade == self.n:
            total = custo + self.c[atual][0]
            if total < self.melhor_custo:
                self.melhor_custo = total
                self.melhor_tour = list(self.caminho)
                self.melhorias += 1
            return

        linha = self.c[atual]
        for proximo in self.ordem[atual]:
            if self.visitado[proximo]:
                continue
            novo_custo = custo + linha[proximo]
            self._entrar(proximo)
            if self._limite(proximo, novo_custo) >= self.melhor_custo:
                self.nos_podados += 1
            else:
                self._dfs(proximo, novo_custo, profundidade + 1)
            self._sair(proximo)

    def _entrar(self, v: int) -> None:
        self.visitado[v] = True
        self.caminho.append(v)
        self.soma_m12_livres -= self.m12[v]

    def _sair(self, v: int) -> None:
        self.visitado[v] = False
        self.caminho.pop()
        self.soma_m12_livres += self.m12[v]

    # ------------------------------------------------------ limite inferior
    def _limite(self, atual: int, custo: int) -> int:
        self.chamadas_limite += 1
        if not self.cronometrar_limite:
            return self._calcular_limite(atual, custo)
        inicio = time.perf_counter()
        valor = self._calcular_limite(atual, custo)
        self.tempo_limite += time.perf_counter() - inicio
        return valor

    def _calcular_limite(
        self, atual: int, custo: int, ignorar_pre_filtro: bool = False
    ) -> int:
        soma = self.soma_m12_livres + self.m1[atual] + self.m1[0]
        barato = custo + (soma + 1) // 2
        if self.tipo_limite == "dois_menores":
            return barato
        if barato >= self.melhor_custo and not ignorar_pre_filtro:
            return barato
        livres = [v for v in range(self.n) if not self.visitado[v]]
        if not livres:
            return custo + self.c[atual][0]
        indices = np.array(livres)
        resto = (
            float(self.penalizados[atual, indices].min())
            + float(self.penalizados[indices, 0].min())
            + self._custo_agm(indices)
            - 2 * float(self.pi[indices].sum())
            - float(self.pi[atual] + self.pi[0])
        )
        caro = custo + math.ceil(resto - _FOLGA_PONTO_FLUTUANTE)
        return max(barato, caro)

    def _custo_agm(self, indices: npt.NDArray[np.intp]) -> float:
        """Custo da AGM (custos penalizados) do subgrafo induzido, O(k²).

        Usa só linhas da matriz restritas a ``indices``; não copia a submatriz.
        """
        k = len(indices)
        if k <= 1:
            return 0.0
        distancia = self.penalizados[indices[0], indices].copy()
        na_arvore = np.zeros(k, dtype=bool)
        na_arvore[0] = True
        distancia[0] = np.inf
        total = 0.0
        for _ in range(k - 1):
            j = int(np.argmin(distancia))
            total += float(distancia[j])
            na_arvore[j] = True
            np.minimum(distancia, self.penalizados[indices[j], indices], out=distancia)
            distancia[na_arvore] = np.inf
        return total


def _rotacionar_para_zero(tour: Sequence[int]) -> Tour:
    i = list(tour).index(0)
    return list(tour[i:]) + list(tour[:i])


def branch_and_bound(
    custos: MatrizCustos,
    tour_inicial: Sequence[int] | None = None,
    limite_tempo_s: float = 1800.0,
    limite_inferior: LimiteInferior = "held_karp",
    medidor: Medidor = MEDIDOR_NULO,
) -> ResultadoBnB:
    """Atalho: cria o ``BranchAndBound`` e resolve com o warm-start dado."""
    bnb = BranchAndBound(custos, limite_tempo_s, limite_inferior)
    return bnb.resolver(tour_inicial, medidor)
