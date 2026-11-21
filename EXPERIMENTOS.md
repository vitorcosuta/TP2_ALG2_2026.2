# Plano de experimentação

Este documento diz **o que medir em cada algoritmo** e **como cada medida vira
uma seção do relatório**. Ele segue a especificação do TP2 (seções
"Instrumentação, Modelagem e Avaliação", "Relatório" e "O que entregar?").
A organização do código em camadas (coleta → dados → análise) está em
[ESTRUTURA.md](ESTRUTURA.md).

> **Prazo:** entrega em **24/11/2026 às 23h59**. Cada execução do B&B pode
> levar até 30 min. Com ~14 instâncias × 5 warm-starts, o pior caso passa de
> **35 h de máquina**. As execuções do B&B devem começar assim que ele estiver
> pronto, divididas entre as máquinas do grupo.

---

## 1. Base comum

### Instâncias

- De **12 a 16** instâncias, com **80% delas com N ≤ 40**. O restante fica
  entre **45 e 75** (faixa de estresse, em que o B&B deve estourar o tempo).
- **Conjunto final** (`experimentos/instancias.py`), com 12 instâncias:
  - N ≤ 40 (9): burma14, ulysses16, gr17, gr21, ulysses22, gr24, fri26,
    bayg29, bays29;
  - faixa de estresse (3): att48, berlin52, st70.
- **Desvio da especificação: 75% com N ≤ 40, e não 80%.** As 9 instâncias
  acima são todas as que temos com N ≤ 40 entre as instâncias da TSPLIB com
  N em [10, 75]. Com o mínimo de 12 instâncias, 80% exigiria 10 pequenas; e
  ficar com 11 (9 pequenas = 82%) violaria o mínimo de 12. **O relatório deve
  justificar esse desvio pela falta de instâncias pequenas disponíveis.**
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
| `execucoes` | algoritmo × instância | custo, gap, tempo total, pico de memória, status (OK/NA). Os 5 algoritmos construtivos (coluna `tipo`: `heuristica` ou `aproximativo`) |
| `subrotinas` | algoritmo × instância × etapa | saída de `Medicao.linhas()` (tempo, memória, chamadas, coletas de GC), para os mesmos 5 algoritmos |
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

#### Script de execução (`experimentos/branch_and_bound.py`)

```bash
python -m experimentos.branch_and_bound                  # 12 instâncias × 5 estratégias
python -m experimentos.branch_and_bound --processos 4    # 4 combinações em paralelo
python -m experimentos.branch_and_bound --instancias att48 st70 \
    --saida resultados/bnb_maquina2.csv                   # divisão entre máquinas
python -m experimentos.branch_and_bound --consolidar     # gera otimos_bnb.csv (só sob demanda)
```

- **Retomável:** cada linha é gravada assim que a execução termina; ao
  reiniciar, o que já está no CSV é pulado. Uma execução é identificada por
  instância, estratégia, limite inferior **e limite de tempo**. Para refazer,
  use outra `--saida` ou apague o arquivo.
- **Um arquivo, um limite de tempo:** se a `--saida` já tem execuções com
  outro limite (ex.: um teste rápido com `--limite-tempo 5`), o script para
  com erro em vez de misturar. Assim um teste nunca é confundido com a
  execução oficial.
- **`otimos_bnb.csv` só com `--consolidar`:** nenhuma execução gera esse
  arquivo sozinha, para que nenhuma máquina produza uma versão parcial.
- Ordem: instâncias da menor para a maior. Estimativa: ~8 h de máquina com
  1 processo (as 3 grandes × 5 estratégias × 30 min); bem menos em paralelo.
- `--estrategias sem_warm_start` roda o B&B sem limite superior inicial (fora
  do padrão; ver o passo 3b da seção 6.3).

**`resultados/bnb.csv`**: uma linha por (instância × estratégia).

| Coluna | Significado |
|---|---|
| `conjunto` | `principal` (as 12 de `SELECIONADAS`), `extra` (TSPLIB fora delas, só para a regressão) ou `aleatorio` |
| `instancia`, `semente`, `n`, `estrategia` | identificação da combinação (`semente` só nas aleatórias) |
| `limite_inferior`, `limite_tempo_s` | parâmetros do B&B (`held_karp`, 1800) |
| `origem_pontos` | só no fecho convexo: seção de onde vieram os pontos, ou `MDS` |
| `custo_inicial` | custo do tour da heurística = limite superior inicial (vazio sem warm-start) |
| `tempo_heuristica_s` | mediana de 10 execuções da heurística |
| `status` | `OK` (ótimo provado) ou `NA` (estourou o tempo) |
| `custo`, `tour` | melhor tour achado (o ótimo se `OK`); tour como `0-5-3-…` |
| `limite_raiz` | limite inferior de Held-Karp na raiz (cota inferior do ótimo) |
| `nos_expandidos`, `nos_podados` | nós visitados e filhos descartados pelo limite |
| `profundidade_maxima`, `melhorias` | maior profundidade e quantas vezes o incumbente melhorou |
| `chamadas_limite_inferior` | nº de cálculos de limite inferior |
| `tempo_bnb_s` | penalidades + busca |
| `tempo_penalidades_s`, `tempo_busca_s` | as duas fases do B&B |
| `tempo_limite_inferior_s` | parte da busca gasta calculando limites (profiling) |
| `tempo_total_s` | heurística + B&B (trade-off do E2) |
| `pico_memoria_bytes`, `pico_memoria_busca_bytes` | execução separada com `tracemalloc`, só se `OK` em ≤ 120 s; vazio caso contrário |
| `processador`, `nucleos_fisicos`, `nucleos_logicos` | modelo do processador e nº de núcleos (lógicos incluem hyper-threading; físicos vazio se o sistema não informar) |
| `python`, `data` | auditoria: versão do Python e quando rodou |

**`resultados/otimos_bnb.csv`**: uma linha por instância (principais e
extras; as aleatórias ficam de fora), juntando todos os `bnb*.csv` da pasta.
Gerado só com `--consolidar`, no passo 5. É o ótimo de referência para os
gaps de todos os algoritmos.

| Coluna | Significado |
|---|---|
| `otimo_provado` | alguma estratégia terminou com `OK` |
| `melhor_custo` | o ótimo (se provado) ou o melhor tour achado (cota superior) |
| `maior_limite_raiz` | melhor cota inferior; se não provado, ótimo ∈ [este, `melhor_custo`] |
| `execucoes`, `execucoes_ok` | quantas combinações rodaram e quantas terminaram |

### 2.2 Heurísticas de warm-start

São três: **trivial** (1 → 2 → … → n → 1), **vizinho mais próximo** e
**inserção por fecho convexo**.

| Medida | Usada em |
|---|---|
| Custo do tour (= limite superior inicial do B&B) e gap | E2 |
| Tempo | E2 (trade-off), E1 |

Implementação: `tsp/heuristicas.py` (`trivial`, `vizinho_mais_proximo`,
`insercao_fecho_convexo`), com tour começando em 0.

- **Fecho convexo nas instâncias sem coordenadas** (gr17, gr21, gr24, gr48,
  fri26, hk48, brazil58, swiss42): os pontos são reconstruídos da matriz por
  **MDS clássico** (qualidade ≥ 0,85). As coordenadas só definem o ciclo
  inicial; a inserção usa os custos reais. O tempo do MDS entra na etapa `mds`.
  Isso deve constar no relatório.
- **Atenção no E2:** em dantzig42 a ordem do arquivo já está a 0,7% do ótimo,
  então o "trivial" não é um warm-start ruim nessa instância.
- Gap contra o limite de Held-Karp (todas as instâncias): inserção 0–15%
  (melhor que o Christofides em quase todas), vizinho mais próximo 5–51%,
  trivial 1–415%.

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
- **Instâncias: só as com ótimo provado pelo B&B** (`otimo_provado` em
  `otimos_bnb.csv`). Espera-se 13: as 9 principais com N ≤ 40 e as 4 extras
  (swiss42, dantzig42, hk48, eil51). Ficam de fora as instâncias em que o B&B
  estourou o tempo (att48, berlin52, st70), cujo gap é só um intervalo, e as
  que o B&B não roda (gr48, brazil58). Declarar no relatório.
- **Variável resposta:** gap. Opcionalmente, nós expandidos do B&B.
- **Modelo:** regressão múltipla ou árvore de decisão, com as features da
  seção 4.
- **Atenção:** com 13 instâncias e 5 ou mais features o modelo sobreajusta.
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

## 6. Plano de execução dos scripts

### 6.1 Princípio: o que pode ser paralelizado e onde rodar

Cada execução roda em **um núcleo** (Python puro), então um processador mais
rápido por núcleo diminui o tempo de cada execução; mais núcleos só ajudam
rodando várias execuções ao mesmo tempo (`--processos`). Daí duas regras:

1. **Tudo que vira tempo comparado (regressões do E1, trade-off do E2) roda na
   mesma máquina, em sequência, com a máquina ociosa.** Processos simultâneos
   disputam cache e memória e inflam os tempos.
2. **Paralelizar só o que é longo e cujo tempo não é a medida:** o B&B nas
   instâncias grandes, que estoura os 30 min de qualquer forma.

Custo, tour, gap e nós expandidos (quando o status é OK) **não dependem da
máquina**: são determinísticos.

### 6.2 Inventário

Durações medidas num Intel i5-1135G7 (4 núcleos físicos).

| Script | Gera | Duração | Paralelizar? | Onde rodar |
|---|---|---|---|---|
| `experimentos.features` | `features.csv` | segundos | não precisa | qualquer máquina |
| `experimentos.aproximados_tsplib` (3 heurísticas + 2 aproximados) | `execucoes.csv`, `subrotinas.csv` e as versões `_sem_gc` | ~4 s | **não** (tempo é a medida) | máquina de referência |
| `experimentos.aproximados_aleatorios` (3 heurísticas + 2 aproximados) | `subrotinas_aleatorias_e1.csv` (a regressão e os gráficos saem de `analises.e1`) | ~6–7 min (com N=1000, Christofides leva ~10 s e a inserção ~3 s por execução) | **não** | máquina de referência |
| `experimentos.branch_and_bound`, 9 pequenas | `bnb_pequenas.csv` | ~2 min | **não** (`--processos 1`) | máquina de referência |
| `experimentos.branch_and_bound`, 4 extras (só Christofides) | `bnb_extras.csv` | ~2 min | **não** | máquina de referência |
| `experimentos.bnb_aleatorias` (45 instâncias, só Christofides) | `bnb_aleatorias.csv` | ~15–30 min | **não** | máquina de referência |
| `experimentos.branch_and_bound`, 3 grandes | `bnb_<instancia>.csv` | 5 × 30 min = 2,5 h por instância | **sim** | qualquer máquina, mas as 5 estratégias de uma instância juntas |
| `experimentos.branch_and_bound --consolidar` | `otimos_bnb.csv` | segundos | não | quem juntar os resultados (passo 5) |
| `experimentos.verificar` | nada (só confere) | segundos | não | quem juntar os resultados (passo 5) |

**Máquina de referência:** a mais rápida do grupo, de preferência um desktop.
Ela roda tudo o que é medido em tempo; o relatório informa o processador, a
RAM e a versão do Python dela (e das outras, para o B&B grande).

### 6.3 Passo a passo

**Passo 0: preparação (todas as máquinas).**

```bash
git pull                      # todos no mesmo commit
make install                  # cria o .venv e instala as mesmas versões
source .venv/bin/activate     # sem isto, "python" pode ser o do sistema
make check                    # tudo passando antes de medir
```

Todos os comandos abaixo supõem o ambiente ativado (`source
.venv/bin/activate` em cada terminal novo). Notebook na tomada, em modo
desempenho, sem programas pesados abertos (o navegador conta). Não usar a
máquina para outra coisa enquanto mede.

**Testes rápidos antes de rodar de verdade:** sempre com uma `--saida` fora
de `resultados/`, por exemplo:

```bash
python -m experimentos.branch_and_bound --instancias burma14 --limite-tempo 5 \
    --saida /tmp/teste_bnb.csv
```

Se um teste for gravado por engano no arquivo oficial, a execução real para
com erro (limite de tempo diferente) em vez de aproveitar o resultado do
teste; nesse caso, apague o arquivo e rode de novo.

**Passo 1: features (qualquer máquina).**

```bash
python -m experimentos.features --mds
```

**Passo 2: aproximados (máquina de referência, um script de cada vez).**

```bash
python -m experimentos.aproximados_tsplib
python -m experimentos.aproximados_aleatorios
```

**Passo 3: B&B nas 9 instâncias pequenas (máquina de referência, sequencial).**
Os tempos destas execuções entram no E1 (regressão do B&B) e no trade-off do
E2, por isso `--processos 1` e nada mais rodando.

```bash
python -m experimentos.branch_and_bound --processos 1 \
    --instancias burma14 ulysses16 gr17 gr21 ulysses22 gr24 fri26 bayg29 bays29 \
    --saida resultados/bnb_pequenas.csv
```

**Passo 3b: conjunto de regressão do B&B (máquina de referência, sequencial).**
Só para o E1 (ver 6.6). Roda **só com o warm-start do Christofides**, para que
a regressão compare instâncias, não estratégias.

```bash
# 4 instâncias extras da TSPLIB (conjunto = extra), ~2 min
python -m experimentos.branch_and_bound --processos 1 \
    --instancias swiss42 dantzig42 hk48 eil51 --estrategias christofides \
    --saida resultados/bnb_extras.csv

# 45 instâncias aleatórias: N = 10, 15, …, 50 × 5 sementes (conjunto = aleatorio)
python -m experimentos.bnb_aleatorias --processos 1 \
    --tamanhos 10 15 20 25 30 35 40 45 50 --sementes 5 --estrategias christofides \
    --saida resultados/bnb_aleatorias.csv
```

Os valores acima são os padrões; estão explícitos para ficar claro o que roda.
As aleatórias são geradas de forma determinística a partir de (N, semente):
qualquer máquina gera as mesmas instâncias.

Opcional: `--estrategias sem_warm_start` nas 12 principais, numa `--saida`
própria (ex.: `resultados/bnb_sem_warm_start.csv`). É o B&B a frio de fato,
sem nenhum limite superior inicial; o "trivial" já é um warm-start ruim nas
principais (gap de 22% a 188%), então este nível só mostra o quanto até um
limite ruim ajuda. O `verificar` não exige essas linhas.

**Passo 4: B&B nas 3 instâncias grandes (pode ser em paralelo).** O tempo
aqui é sempre ~1800 s (NA); o que se compara são os nós expandidos e o melhor
tour achado, que dependem da velocidade da máquina. Por isso **as 5
estratégias de uma mesma instância rodam na mesma máquina e nas mesmas
condições**. Duas opções, ambas com ~2,5 h de relógio:

- **A (recomendada): uma instância por pessoa**, em sequência:
  ```bash
  python -m experimentos.branch_and_bound --processos 1 \
      --instancias st70 --saida resultados/bnb_st70.csv
  ```
- **B: uma máquina só**, com as 3 instâncias em paralelo (no máximo um
  processo por núcleo físico, deixando um livre):
  ```bash
  python -m experimentos.branch_and_bound --processos 3 \
      --instancias att48 berlin52 st70 --saida resultados/bnb_grandes.csv
  ```
  Com 3 processos sempre ativos, todas as execuções disputam a máquina por
  igual; declarar no relatório.

Nunca rodar o passo 4 **na máquina de referência ao mesmo tempo** que os
passos 2 e 3. Se cair, é só rodar o mesmo comando: o script retoma.

**Passo 5: juntar e conferir os resultados.** Cada um commita só os arquivos
que gerou (todos têm nomes diferentes, então não há conflito no git):

| Quem | Commita em `resultados/` |
|---|---|
| Quem rodou o passo 1 | `features.csv` |
| Máquina de referência (passos 2, 3 e 3b) | `execucoes*.csv`, `subrotinas*.csv`, `subrotinas_aleatorias_e1.csv`, `bnb_pequenas.csv`, `bnb_extras.csv`, `bnb_aleatorias.csv` |
| Cada máquina do passo 4 | o seu `bnb_<instancia>.csv` (ou `bnb_grandes.csv`) |

Ninguém commita `otimos_bnb.csv` durante as execuções (nenhuma execução o
gera). Depois que todos commitarem, **uma pessoa** roda:

```bash
git pull                                              # traz os CSVs de todos
python -m experimentos.branch_and_bound --consolidar  # gera otimos_bnb.csv completo
python -m experimentos.verificar                      # confere se não falta nada
git add resultados/otimos_bnb.csv && git commit -m "consolidar ótimos do B&B"
```

O `verificar` compara `resultados/` com este plano e lista, por passo, o que
falta, o que está duplicado, execuções com parâmetros diferentes dos oficiais
(limite de 1800 s, Held-Karp), um `otimos_bnb.csv` desatualizado e tempos
comparados vindos de mais de um processador. Só seguir para as análises com
"Coleta completa." (código de saída 0).

**Passo 6: análises (camada `analises/`, qualquer máquina, segundos).** Leem
só `resultados/` e gravam em `relatorio/` (ver [ESTRUTURA.md](ESTRUTURA.md)).

```bash
python -m analises.e1        # pronto: regressão e gráficos dos aproximados
```

A implementar:
- Preencher `gap_bnb` dos aproximados e das heurísticas com `otimos_bnb.csv`.
- E1 do B&B: regressão de `tempo_bnb_s` contra aⁿ, n²·2ⁿ e n! nas execuções OK.
- E2: tabela instância × estratégia, correlação gap inicial × nós, trade-off
  `tempo_total_s` contra o trivial, ponto de colapso.
- E3: modelo do gap contra as features com leave-one-out.

### 6.4 Divisão sugerida e cronograma

| Quem | 21/11 | 22/11 | 23/11 |
|---|---|---|---|
| Pessoa 1 (máquina de referência) | passos 0–3b, depois passo 4 com att48 | E1 e E2 | revisão |
| Pessoa 2 | passo 4 com berlin52 | E3 | conclusões |
| Pessoa 3 | passo 4 com st70; passo 1 | seções 1 e 2 do relatório | anexo de prompts |

Entrega: 24/11 às 23h59. O dia 24 fica de folga para imprevistos.

### 6.5 Observações

- O `aproximados_tsplib` roda as 18 instâncias da pasta, não só as 12
  principais. No E1 todas podem entrar; no E3 só as 13 com ótimo provado
  (seção 3). O relatório deve dizer quais instâncias entram em cada análise.
- Todos os CSVs de coleta têm as colunas `processador`, `nucleos_fisicos`,
  `nucleos_logicos` e `python`: o hardware da máquina de referência para o
  relatório sai dos próprios dados.

### 6.6 Regressão do B&B em três níveis

A regressão do B&B (tempo e nós expandidos × N, contra aⁿ, n²·2ⁿ e n!) é
feita em três níveis, para comparar a qualidade do ajuste conforme a amostra
cresce. Os níveis são separados pela coluna **`conjunto`** dos CSVs do B&B, e
cada linha leva o seu rótulo desde a coleta: nenhum cruzamento depende do nome
do arquivo.

| Nível | Filtro (`conjunto`) | Arquivos | Pontos que terminam (OK) | Faixa de N |
|---|---|---|---|---|
| 1. Principal | `principal` | `bnb_pequenas.csv`, `bnb_<grande>.csv` | 9 | 14–29 |
| 2. Principal + extras | `principal` ou `extra` | os de cima + `bnb_extras.csv` | 13 | 14–51 |
| 3. Aleatórias | `aleatorio` | `bnb_aleatorias.csv` | ~45 | 10–50 |

Regras comuns aos três níveis:

- Só linhas com `estrategia == "christofides"`. As extras e as aleatórias só
  rodam com ela, e misturar estratégias confundiria o efeito de N com o do
  warm-start.
- Só linhas com `status == "OK"` entram no ajuste. As NA (≥ 1800 s) aparecem
  nos gráficos como pontos censurados, com marcador próprio.
- No nível 3, ajustar também a mediana por N das 5 sementes. Instâncias com o
  mesmo N chegam a ter tempos 10× diferentes.
- O ajuste é em escala log, log(y) contra log(modelo), e reporta R² e RMSE.
  Ajustar tanto `tempo_bnb_s` quanto `nos_expandidos` (este não depende da
  máquina).
- Nos gráficos, uma cor por nível, ou um painel por nível, sempre com a
  legenda do nível.
- Todos os tempos usados vêm da máquina de referência (passos 3 e 3b).

