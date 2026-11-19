"""Emparelhamento perfeito de custo mínimo pelo algoritmo Blossom de Edmonds.

Implementa a versão primal-dual em O(n³) de Gabow e Galil para emparelhamento
de peso **máximo**, seguindo a estrutura da implementação de referência de
Joris van Rantwijk (também usada pelo NetworkX). O emparelhamento perfeito de
custo **mínimo** sai dela por redução: com cardinalidade máxima forçada, todo
emparelhamento perfeito tem k/2 arestas, então maximizar ``Σ (C − c(u, v))``
equivale a minimizar ``Σ c(u, v)``.

Nomenclatura (como na referência):

- Cada aresta ``k = (i, j)`` tem duas extremidades: ``2k`` (lado ``i``) e
  ``2k + 1`` (lado ``j``); ``extremidade[p]`` é o vértice da extremidade ``p``
  e ``p ^ 1`` é a extremidade oposta da mesma aresta.
- Índices ``0..n−1`` são vértices; ``n..2n−1`` são blossoms (flores).
- Rótulos: 0 = livre, 1 = S (externo), 2 = T (interno).
- ``dual[v]`` é a variável dual do vértice; ``dual[b]`` (b ≥ n) a da flor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain

import numpy as np

from tsp.instancia import MatrizCustos

Aresta = tuple[int, int, int]

_LIVRE, _S, _T = 0, 1, 2
_MARCA = 4  # bit temporário usado por _buscar_flor


class _Blossom:
    """Estado do algoritmo de emparelhamento de peso máximo."""

    def __init__(self, n: int, arestas: Sequence[Aresta]) -> None:
        self.n = n
        self.arestas = arestas
        m = len(arestas)
        maior_peso = max(0, max(w for _, _, w in arestas))
        self.extremidade = [arestas[p // 2][p % 2] for p in range(2 * m)]
        self.vizinhos: list[list[int]] = [[] for _ in range(n)]
        for k, (i, j, _) in enumerate(arestas):
            self.vizinhos[i].append(2 * k + 1)
            self.vizinhos[j].append(2 * k)

        self.par = [-1] * n  # extremidade casada com o vértice (−1 = livre)
        self.rotulo = [_LIVRE] * (2 * n)
        self.fim_rotulo = [-1] * (2 * n)
        self.na_flor = list(range(n))  # flor de mais alto nível de cada vértice
        self.flor_pai = [-1] * (2 * n)
        self.flor_filhos: list[list[int]] = [[] for _ in range(2 * n)]
        self.flor_base = list(range(n)) + [-1] * n
        self.flor_extremidades: list[list[int]] = [[] for _ in range(2 * n)]
        self.melhor_aresta = [-1] * (2 * n)
        self.flor_melhores: list[list[int] | None] = [None] * (2 * n)
        self.flores_livres = list(range(n, 2 * n))
        self.dual = [maior_peso] * n + [0] * n
        self.permitida = [False] * m
        self.fila: list[int] = []

    # ------------------------------------------------------------ auxiliares
    def _folga(self, k: int) -> int:
        i, j, w = self.arestas[k]
        return self.dual[i] + self.dual[j] - 2 * w

    def _folhas(self, b: int) -> Iterator[int]:
        if b < self.n:
            yield b
            return
        for t in self.flor_filhos[b]:
            yield from self._folhas(t)

    def _rotular(self, w: int, t: int, p: int) -> None:
        """Rotula ``w`` (e sua flor) com ``t`` alcançado pela extremidade ``p``."""
        b = self.na_flor[w]
        self.rotulo[w] = self.rotulo[b] = t
        self.fim_rotulo[w] = self.fim_rotulo[b] = p
        self.melhor_aresta[w] = self.melhor_aresta[b] = -1
        if t == _S:
            self.fila.extend(self._folhas(b))
        else:
            base = self.flor_base[b]
            self._rotular(self.extremidade[self.par[base]], _S, self.par[base] ^ 1)

    def _buscar_flor(self, v: int, w: int) -> int:
        """Sobe pelas árvores de ``v`` e ``w``; devolve a base da nova flor ou −1."""
        caminho: list[int] = []
        base = -1
        while v != -1 or w != -1:
            b = self.na_flor[v]
            if self.rotulo[b] & _MARCA:
                base = self.flor_base[b]
                break
            caminho.append(b)
            self.rotulo[b] = _S | _MARCA
            if self.fim_rotulo[b] == -1:
                v = -1
            else:
                v = self.extremidade[self.fim_rotulo[b]]
                b = self.na_flor[v]
                v = self.extremidade[self.fim_rotulo[b]]
            if w != -1:
                v, w = w, v
        for b in caminho:
            self.rotulo[b] = _S
        return base

    # ----------------------------------------------------------- flores
    def _criar_flor(self, base: int, k: int) -> None:
        v, w, _ = self.arestas[k]
        bb, bv, bw = self.na_flor[base], self.na_flor[v], self.na_flor[w]
        b = self.flores_livres.pop()
        self.flor_base[b] = base
        self.flor_pai[b] = -1
        self.flor_pai[bb] = b
        filhos: list[int] = []
        extremidades: list[int] = []
        while bv != bb:
            self.flor_pai[bv] = b
            filhos.append(bv)
            extremidades.append(self.fim_rotulo[bv])
            bv = self.na_flor[self.extremidade[self.fim_rotulo[bv]]]
        filhos.append(bb)
        filhos.reverse()
        extremidades.reverse()
        extremidades.append(2 * k)
        while bw != bb:
            self.flor_pai[bw] = b
            filhos.append(bw)
            extremidades.append(self.fim_rotulo[bw] ^ 1)
            bw = self.na_flor[self.extremidade[self.fim_rotulo[bw]]]
        self.flor_filhos[b] = filhos
        self.flor_extremidades[b] = extremidades

        self.rotulo[b] = _S
        self.fim_rotulo[b] = self.fim_rotulo[bb]
        self.dual[b] = 0
        for folha in self._folhas(b):
            if self.rotulo[self.na_flor[folha]] == _T:
                self.fila.append(folha)
            self.na_flor[folha] = b
        self._atualizar_melhores_da_flor(b, filhos)

    def _arestas_candidatas(self, filho: int) -> list[int]:
        """Arestas que saem de ``filho``: as já filtradas, se houver, ou todas."""
        guardadas = self.flor_melhores[filho]
        if guardadas is not None:
            return guardadas
        return [p // 2 for folha in self._folhas(filho) for p in self.vizinhos[folha]]

    def _atualizar_melhores_da_flor(self, b: int, filhos: list[int]) -> None:
        melhor_para = [-1] * (2 * self.n)
        for filho in filhos:
            for k in self._arestas_candidatas(filho):
                i, j, _ = self.arestas[k]
                bj = self.na_flor[i] if self.na_flor[j] == b else self.na_flor[j]
                if bj == b or self.rotulo[bj] != _S:
                    continue
                atual = melhor_para[bj]
                if atual == -1 or self._folga(k) < self._folga(atual):
                    melhor_para[bj] = k
            self.flor_melhores[filho] = None
            self.melhor_aresta[filho] = -1
        melhores = [k for k in melhor_para if k != -1]
        self.flor_melhores[b] = melhores
        self.melhor_aresta[b] = min(melhores, key=self._folga, default=-1)

    def _expandir_flor(self, b: int, fim_de_fase: bool) -> None:
        for s in self.flor_filhos[b]:
            self.flor_pai[s] = -1
            if s < self.n:
                self.na_flor[s] = s
            elif fim_de_fase and self.dual[s] == 0:
                self._expandir_flor(s, fim_de_fase)
            else:
                for folha in self._folhas(s):
                    self.na_flor[folha] = s
        if not fim_de_fase and self.rotulo[b] == _T:
            self._rerotular_flor_t(b)
        self.rotulo[b] = self.fim_rotulo[b] = -1
        self.flor_filhos[b] = []
        self.flor_extremidades[b] = []
        self.flor_base[b] = -1
        self.flor_melhores[b] = None
        self.melhor_aresta[b] = -1
        self.flores_livres.append(b)

    def _sentido(self, b: int, j: int) -> tuple[int, int, int]:
        """Percorre o ciclo da flor pelo lado de comprimento par até o filho 0."""
        if j & 1:
            return j - len(self.flor_filhos[b]), 1, 0
        return j, -1, 1

    def _rerotular_flor_t(self, b: int) -> None:
        """Ao expandir uma flor T no meio da fase, refaz os rótulos dos filhos."""
        filhos, ends = self.flor_filhos[b], self.flor_extremidades[b]
        entrada = self.na_flor[self.extremidade[self.fim_rotulo[b] ^ 1]]
        j, passo, truque = self._sentido(b, filhos.index(entrada))
        p = self.fim_rotulo[b]
        while j != 0:
            self.rotulo[self.extremidade[p ^ 1]] = _LIVRE
            self.rotulo[self.extremidade[ends[j - truque] ^ truque ^ 1]] = _LIVRE
            self._rotular(self.extremidade[p ^ 1], _T, p)
            self.permitida[ends[j - truque] // 2] = True
            j += passo
            p = ends[j - truque] ^ truque
            self.permitida[p // 2] = True
            j += passo
        bv = filhos[j]
        self.rotulo[self.extremidade[p ^ 1]] = self.rotulo[bv] = _T
        self.fim_rotulo[self.extremidade[p ^ 1]] = self.fim_rotulo[bv] = p
        self.melhor_aresta[bv] = -1
        j += passo
        while filhos[j] != entrada:
            bv = filhos[j]
            j += passo
            if self.rotulo[bv] == _S:
                continue
            rotulada = next((v for v in self._folhas(bv) if self.rotulo[v]), None)
            if rotulada is not None:
                self.rotulo[rotulada] = _LIVRE
                base = self.flor_base[bv]
                self.rotulo[self.extremidade[self.par[base]]] = _LIVRE
                self._rotular(rotulada, _T, self.fim_rotulo[rotulada])

    # -------------------------------------------------------- aumento
    def _aumentar_flor(self, b: int, v: int) -> None:
        """Troca as arestas do caminho alternante dentro de ``b`` até ``v``."""
        t = v
        while self.flor_pai[t] != b:
            t = self.flor_pai[t]
        if t >= self.n:
            self._aumentar_flor(t, v)
        i = self.flor_filhos[b].index(t)
        j, passo, truque = self._sentido(b, i)
        filhos, ends = self.flor_filhos[b], self.flor_extremidades[b]
        while j != 0:
            j += passo
            t = filhos[j]
            p = ends[j - truque] ^ truque
            if t >= self.n:
                self._aumentar_flor(t, self.extremidade[p])
            j += passo
            t = filhos[j]
            if t >= self.n:
                self._aumentar_flor(t, self.extremidade[p ^ 1])
            self.par[self.extremidade[p]] = p ^ 1
            self.par[self.extremidade[p ^ 1]] = p
        self.flor_filhos[b] = filhos[i:] + filhos[:i]
        self.flor_extremidades[b] = ends[i:] + ends[:i]
        self.flor_base[b] = self.flor_base[self.flor_filhos[b][0]]

    def _aumentar_emparelhamento(self, k: int) -> None:
        v, w, _ = self.arestas[k]
        for s, p in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = self.na_flor[s]
                if bs >= self.n:
                    self._aumentar_flor(bs, s)
                self.par[s] = p
                if self.fim_rotulo[bs] == -1:
                    break
                bt = self.na_flor[self.extremidade[self.fim_rotulo[bs]]]
                s = self.extremidade[self.fim_rotulo[bt]]
                j = self.extremidade[self.fim_rotulo[bt] ^ 1]
                if bt >= self.n:
                    self._aumentar_flor(bt, j)
                self.par[j] = self.fim_rotulo[bt]
                p = self.fim_rotulo[bt] ^ 1

    # ------------------------------------------------------- fase principal
    def _iniciar_fase(self) -> None:
        self.rotulo = [_LIVRE] * (2 * self.n)
        self.melhor_aresta = [-1] * (2 * self.n)
        self.flor_melhores[self.n :] = [None] * self.n
        self.permitida = [False] * len(self.arestas)
        self.fila = []
        for v in range(self.n):
            if self.par[v] == -1 and self.rotulo[self.na_flor[v]] == _LIVRE:
                self._rotular(v, _S, -1)

    def _examinar(self, v: int, p: int) -> bool:
        """Examina a aresta pela extremidade ``p`` saindo de ``v`` (rótulo S).

        Devolve ``True`` se o emparelhamento foi aumentado.
        """
        k, w = p // 2, self.extremidade[p]
        bw = self.na_flor[w]
        if self.na_flor[v] == bw:
            return False
        folga = 0
        if not self.permitida[k]:
            folga = self._folga(k)
            self.permitida[k] = folga <= 0
        if self.permitida[k]:
            if self.rotulo[bw] == _LIVRE:
                self._rotular(w, _T, p ^ 1)
            elif self.rotulo[bw] == _S:
                base = self._buscar_flor(v, w)
                if base < 0:
                    self._aumentar_emparelhamento(k)
                    return True
                self._criar_flor(base, k)
            elif self.rotulo[w] == _LIVRE:
                self.rotulo[w] = _T
                self.fim_rotulo[w] = p ^ 1
        elif self.rotulo[bw] == _S:
            self._guardar_melhor(self.na_flor[v], k, folga)
        elif self.rotulo[w] == _LIVRE:
            self._guardar_melhor(w, k, folga)
        return False

    def _guardar_melhor(self, b: int, k: int, folga: int) -> None:
        atual = self.melhor_aresta[b]
        if atual == -1 or folga < self._folga(atual):
            self.melhor_aresta[b] = k

    def _deltas_tipo_2(self) -> Iterator[tuple[int, int, int]]:
        """Aresta entre um vértice S e um vértice livre."""
        for v in range(self.n):
            k = self.melhor_aresta[v]
            if self.rotulo[self.na_flor[v]] == _LIVRE and k != -1:
                yield self._folga(k), 2, k

    def _deltas_tipo_3(self) -> Iterator[tuple[int, int, int]]:
        """Aresta entre duas flores S distintas (metade da folga)."""
        for b in range(2 * self.n):
            k = self.melhor_aresta[b]
            if self.flor_pai[b] == -1 and self.rotulo[b] == _S and k != -1:
                yield self._folga(k) // 2, 3, k

    def _deltas_tipo_4(self) -> Iterator[tuple[int, int, int]]:
        """Flor T de nível mais alto cuja dual chegaria a zero."""
        for b in range(self.n, 2 * self.n):
            if (
                self.flor_base[b] >= 0
                and self.flor_pai[b] == -1
                and self.rotulo[b] == _T
            ):
                yield self.dual[b], 4, b

    def _escolher_delta(self) -> tuple[int, int, int]:
        """Escolhe o ajuste dual (tipo, delta, aresta ou flor) da iteração.

        Em empate vence o primeiro candidato, na ordem tipo 2, 3, 4. Sem
        candidatos, o tipo 1 encerra a fase (não há caminho aumentante).
        """
        candidatos = chain(
            self._deltas_tipo_2(), self._deltas_tipo_3(), self._deltas_tipo_4()
        )
        melhor = min(candidatos, key=lambda c: c[0], default=None)
        if melhor is None:
            return 1, max(0, min(self.dual[: self.n])), -1
        delta, tipo, alvo = melhor
        return tipo, delta, alvo

    def _aplicar_delta(self, delta: int) -> None:
        for v in range(self.n):
            rotulo = self.rotulo[self.na_flor[v]]
            if rotulo == _S:
                self.dual[v] -= delta
            elif rotulo == _T:
                self.dual[v] += delta
        for b in range(self.n, 2 * self.n):
            if self.flor_base[b] >= 0 and self.flor_pai[b] == -1:
                if self.rotulo[b] == _S:
                    self.dual[b] += delta
                elif self.rotulo[b] == _T:
                    self.dual[b] -= delta

    def _fase(self) -> bool:
        """Executa uma fase; devolve ``True`` se encontrou caminho aumentante."""
        self._iniciar_fase()
        while True:
            while self.fila:
                v = self.fila.pop()
                if any(self._examinar(v, p) for p in self.vizinhos[v]):
                    return True
            tipo, delta, alvo = self._escolher_delta()
            self._aplicar_delta(delta)
            if tipo == 1:
                return False
            if tipo == 4:
                self._expandir_flor(alvo, fim_de_fase=False)
                continue
            self.permitida[alvo] = True
            i, j, _ = self.arestas[alvo]
            if self.rotulo[self.na_flor[i]] == _LIVRE:
                i = j
            self.fila.append(i)

    def resolver(self) -> list[int]:
        """Devolve ``par``: o vértice casado com cada vértice (−1 se livre)."""
        for _ in range(self.n):
            if not self._fase():
                break
            for b in range(self.n, 2 * self.n):
                if (
                    self.flor_pai[b] == -1
                    and self.flor_base[b] >= 0
                    and self.rotulo[b] == _S
                    and self.dual[b] == 0
                ):
                    self._expandir_flor(b, fim_de_fase=True)
        return [self.extremidade[p] if p >= 0 else -1 for p in self.par]


def emparelhamento_peso_maximo_cardinalidade_maxima(
    n: int, arestas: Sequence[Aresta]
) -> list[int]:
    """Entre os emparelhamentos de cardinalidade máxima, um de peso máximo.

    ``arestas`` são triplas ``(i, j, peso)`` com pesos inteiros. Devolve
    ``par`` com ``par[v]`` = parceiro de ``v`` ou −1.
    """
    if not arestas:
        return [-1] * n
    return _Blossom(n, arestas).resolver()


def emparelhamento_perfeito_minimo(
    custos: MatrizCustos, vertices: Sequence[int]
) -> list[tuple[int, int]]:
    """Emparelhamento perfeito de custo mínimo no subgrafo completo de ``vertices``.

    ``vertices`` deve ter tamanho par. Devolve os pares ``(u, v)`` com ``u < v``
    nos rótulos originais dos vértices. Tempo O(k³), com k = ``len(vertices)``.
    """
    k = len(vertices)
    if k % 2:
        raise ValueError(f"emparelhamento perfeito exige nº par de vértices ({k})")
    if k == 0:
        return []
    sub = custos[np.ix_(vertices, vertices)]
    teto = int(sub.max()) + 1
    arestas = [(i, j, teto - int(sub[i, j])) for i in range(k) for j in range(i + 1, k)]
    par = emparelhamento_peso_maximo_cardinalidade_maxima(k, arestas)
    if -1 in par:
        raise RuntimeError("o Blossom não encontrou emparelhamento perfeito")
    return sorted(
        (min(vertices[i], vertices[j]), max(vertices[i], vertices[j]))
        for i, j in enumerate(par)
        if i < j
    )
