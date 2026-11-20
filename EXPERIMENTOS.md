# Plano de experimentação

Este documento diz **o que medir em cada algoritmo** e **como cada medida vira
uma seção do relatório**. Ele segue a especificação do TP2 (seções
"Instrumentação, Modelagem e Avaliação", "Relatório" e "O que entregar?").

> **Prazo:** entrega em **24/11/2026 às 23h59**. Cada execução do B&B pode
> levar até 30 min. Com ~14 instâncias × 5 warm-starts, o pior caso passa de
> **35 h de máquina**. As execuções do B&B devem começar assim que ele estiver
> pronto, divididas entre as máquinas do grupo.

---

## 1. Base comum

### Instâncias

- De **12 a 16** instâncias, com **80% delas com N ≤ 40**. O restante fica
  entre **45 e 75** (faixa de estresse, em que o B&B deve estourar o tempo).
- Em `TSP_Instances/`, as que têm N ≤ 40 são: burma14, ulysses16, gr17, gr21,
  ulysses22, gr24, fri26, bayg29 e bays29. **O conjunto final ainda precisa ser
  decidido.**
- Todas são lidas com `tsp.instancia.carregar_instancia`, que aplica o fecho
  métrico (Floyd–Warshall): a desigualdade triangular vale em todas.

### Ótimo de referência

- O fecho métrico pode ter mudado o ótimo publicado na TSPLIB. Por isso o
  **gap de todos os algoritmos é medido contra o ótimo encontrado pelo B&B**.
- Se o B&B estourar o tempo, usar o melhor tour encontrado e registrar o gap
  como um limite inferior (gap ≥ x).

### Base de dados (entregável obrigatório)

São quatro tabelas em CSV:

| Tabela | Uma linha por | Colunas principais |
|---|---|---|
| `instancias` | instância | N, tipo de peso, origem das coordenadas, features (seção 4) |
| `execucoes` | algoritmo × instância | custo, gap, tempo total, pico de memória, status (OK/NA) |
| `subrotinas` | algoritmo × instância × etapa | saída de `Medicao.linhas()` (tempo, memória, chamadas, coletas de GC) |
| `bnb` | instância × estratégia de warm-start | limite superior inicial, nós expandidos, nós podados, profundidade máxima, tempo, status |

### Repetições

- Aproximados e heurísticas são determinísticos e levam milissegundos. Rodar
  **10 vezes** e usar a **mediana** do tempo.
- O B&B roda **uma vez** por combinação de instância e estratégia.

### Mecanismo de medição (`tsp/medicao.py`)

```python
def algoritmo(custos, medidor: Medidor = MEDIDOR_NULO):
    with medidor.etapa("agm"):
        ...


m = medir(lambda md: algoritmo(custos, medidor=md), repeticoes=10)
m.linhas()  # uma linha por etapa → tabela `subrotinas`
```

- **Tempo e memória são medidos em execuções separadas.** O `tracemalloc`
  deixou o Christofides 2,3× mais lento, então medir os dois juntos distorceria
  o tempo.
- Etapas podem ser aninhadas (`total/emparelhamento`). Etapas repetidas
  acumulam tempo e chamadas.
- `desativar_gc=True` desliga o garbage collector durante a medição, para
  comparar com e sem coleta.
- **Não usar `etapa` em laços quentes.** Cada etapa custa ~5 µs. No B&B, usar
  só em fases grandes e contar nós com inteiros simples.

---

## 2. O que medir em cada algoritmo

### 2.1 Branch-and-Bound (exato)

Executado **5 vezes por instância**, uma por estratégia de warm-start.

Implementação: `tsp/branch_and_bound.py`. Tudo abaixo está em
`ResultadoBnB`, inclusive quando o tempo estoura (status NA).

| Medida (campo de `ResultadoBnB`) | Usada em |
|---|---|
| `nos_expandidos`, `nos_podados`, `profundidade_maxima`, `melhorias` | E2 |
| `custo_inicial` (limite superior do warm-start), `limite_raiz` | E2 |
| `tempo_s`, `tempo_penalidades_s`, `tempo_limite_inferior_s` | E1, E2 |
| Pico de memória (via `medir`, etapas `penalidades` e `busca`) | E1 (cópias de matriz, GC) |
| `custo`, `tour`, `status` (OK ou NA após 30 min) | ótimo de referência; ponto de colapso |

- O limite inferior padrão é o de **Held-Karp** (AGM com penalidades
  calculadas na raiz). `dois_menores` e `agm` continuam disponíveis e podem
  ser comparados no relatório (seção 2: "por que um limite falha").
- `tempo_limite_inferior_s` só é preenchido com `cronometrar_limite=True`,
  que custa ~0,1 µs por chamada.
- Teste com 60 s e warm-start do Christofides: todas as instâncias com
  N ≤ 42 e eil51 foram resolvidas; att48, berlin52 e st70 estouraram.

### 2.2 Heurísticas de warm-start

São três: **trivial** (1 → 2 → … → n → 1), **vizinho mais próximo** e
**inserção por fecho convexo**.

| Medida | Usada em |
|---|---|
| Custo do tour (= limite superior inicial do B&B) e gap | E2 |
| Tempo | E2 (trade-off), E1 |

A inserção por fecho convexo **só roda em instâncias com coordenadas**. Nas
8 sem pontos (gr17, gr21, gr24, gr48, fri26, hk48, brazil58, swiss42) ela fica
NA. Isso deve constar no relatório.

### 2.3 Twice-around-the-tree (`tsp/twice_around_the_tree.py`)

| Medida | Usada em |
|---|---|
| Tempo das etapas `agm`, `pre_ordem`, `atalhos`, `custo` | E1 (Prim deve se ajustar a O(n²)) |
| Custo e gap | E3, E2 (como warm-start) |
| Razão custo ÷ custo da AGM | quanto o atalho economiza em relação ao limite de 2 |

### 2.4 Christofides (`tsp/christofides.py`)

| Medida | Usada em |
|---|---|
| Tempo das etapas `agm`, `impares`, `emparelhamento`, `euleriano`, `atalhos`, `custo` | E1 |
| **k = número de vértices de grau ímpar na AGM** | o Blossom é O(k³): ajustar contra k, não contra N |
| Pico de memória | E1 |
| Custo e gap | E3, E2 (como warm-start) |

Primeira medição (st70): o **emparelhamento ocupa 91% do tempo** e quase todo
o pico de memória. Esse resultado vai para a seção 2 do relatório.

---

## 3. Experimentos

### E1. Complexidade empírica e profiling

*Relatório: "modelos de regressão do comportamento assintótico".*

- **Algoritmos:** todos. Para os aproximados, também cada etapa.
- **Como fazer:** ajustar tempo × N (ou × k) com `scipy.optimize.curve_fit`
  a modelos candidatos e reportar **R² e RMSE** de cada um:
  - B&B: aⁿ, n²·2ⁿ, n!
  - Prim: n²
  - Blossom: k³
  - demais etapas: n, n log n, n²
- Fazer o mesmo com o **pico de memória**.
- **Discutir:** o melhor modelo bate com a teoria? Se não, por quê (overhead
  do Python, cópias, GC)? Comparar tempos com e sem `desativar_gc`.
- **Atenção:** 12 a 16 valores de N não bastam para distinguir n² de n³. Só
  neste experimento, acrescentar **instâncias aleatórias** (pontos uniformes
  no plano, N de 10 a 500) para os aproximados.

### E2. Impacto do warm-start na árvore de busca

*Relatório: "eficiência da poda através das estratégias de warm-start".*

- **Algoritmo:** B&B, com cada uma das 5 estratégias (trivial, vizinho mais
  próximo, fecho convexo, Twice-around-the-tree, Christofides).
- **Tabela central:** instância × estratégia → limite superior inicial, gap do
  limite, tempo da heurística, nós expandidos, nós podados, tempo do B&B,
  status.
- **Perguntas a responder:**
  1. Um limite superior melhor reduz quantos nós? Há correlação entre o gap do
     limite e os nós expandidos?
  2. **Trade-off:** tempo da heurística + tempo do B&B é menor que o B&B a
     frio (trivial)? Em que casos não é?
  3. Em quais instâncias o B&B colapsa (NA) e a partir de qual N?
- **Arguição:** podem perguntar por que o limite inferior falhou numa
  instância específica. Guardar os dados por instância.

### E3. Análise de sensibilidade

*Relatório: "peso das features no desempenho/qualidade da solução".*

- **Algoritmos:** Twice-around-the-tree e Christofides. Opcionalmente também
  vizinho mais próximo e fecho convexo.
- **Variável resposta:** gap. Opcionalmente, nós expandidos do B&B.
- **Modelo:** regressão múltipla ou árvore de decisão, com as features da
  seção 4.
- **Atenção:** com ~14 instâncias e 5 ou mais features o modelo sobreajusta.
  Usar poucas features, validação **leave-one-out** e declarar a limitação.
  Para ampliar a amostra, gerar **instâncias adversariais** (pontos em
  clusters, em faixas estreitas, em círculo).

---

## 4. Features das instâncias

Calculadas uma vez por instância, para a tabela `instancias`.

| Feature | Precisa de coordenadas? |
|---|---|
| Razão de aspecto da caixa que envolve os pontos | sim |
| Proporção de pontos no fecho convexo | sim |
| Entropia espacial / densidade de clusters | sim |
| Grau máximo da AGM e k (nº de vértices de grau ímpar) | não |
| Coeficiente de variação da distância ao vizinho mais próximo | não |
| Fração de pares alterados pelo fecho métrico | não (sugestão: mede quão "não métrica" a instância era) |

- Nas 8 instâncias sem coordenadas, as três primeiras ficam NA. O E3 precisa
  de um modelo só com as features que não dependem de coordenadas, ou de mais
  instâncias com pontos.
- Em bayg29, bays29 e dantzig42 os pontos vêm da `DISPLAY_DATA_SECTION`. Eles
  têm correlação ≥ 0,99 com os custos, mas em outra escala. Features que só
  dependem da forma (razão de aspecto, fecho convexo, entropia) valem; isso
  deve ser explicado no relatório.

---

## 5. Correspondência com o relatório

| Seção do relatório | Conteúdo |
|---|---|
| 2. Implementação, profiling e auditoria | estruturas de dados (matriz de custos, fecho métrico, Prim com vetor, Blossom próprio), escolha da estratégia da árvore do B&B (profundidade × best-first), o que o profiling do E1 revelou |
| 3. Experimentos: regressão | E1 |
| 3. Experimentos: poda | E2 |
| 3. Experimentos: sensibilidade | E3 |
| 4. Conclusões | resumo de E1, E2 e E3 |
| Anexo | **histórico de prompts das LLMs (obrigatório)** |

Entregáveis: código-fonte, a base de dados (seção 1), o relatório em PDF e o
fonte LaTeX (padrão SBC), via GitHub.

---

## 6. Ordem de execução

1. **Feito:** leitor de instâncias, AGM, Twice-around-the-tree, Christofides,
   mecanismo de medição, Branch-and-Bound.
2. **Próximo:** as três heurísticas de warm-start (trivial, vizinho mais
   próximo, fecho convexo). **Disparar as execuções do B&B no mesmo dia.**
3. **Enquanto o B&B roda:** script que gera os CSVs, E1 dos aproximados,
   features, seções 1 e 2 do relatório.
4. **Com os resultados do B&B:** gaps, E2, E3 e conclusões.
