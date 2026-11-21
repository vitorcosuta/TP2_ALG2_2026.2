# Estrutura do projeto

Este documento descreve **como o código está organizado em camadas**, o que
cada camada pode fazer e como os dados passam de uma para a outra. O plano do
que medir está em [EXPERIMENTOS.md](EXPERIMENTOS.md). Lá também estão a ordem
e as máquinas de cada execução.

---

## 1. Visão geral

A biblioteca `tsp/` com os algoritmos fica na base. Acima dela há três
camadas, e cada uma só conversa com a seguinte por meio de **arquivos CSV**:

```
                ┌─────────────────────────────────────────────┐
  base          │ tsp/            algoritmos e medição         │
                └──────────────────────┬──────────────────────┘
                                       │ importa
                ┌──────────────────────▼──────────────────────┐
  camada 1      │ experimentos/   COLETA: roda os algoritmos   │  máquina de referência,
                │                 e grava CSV                  │  minutos a horas
                └──────────────────────┬──────────────────────┘
                                       │ escreve CSV
                ┌──────────────────────▼──────────────────────┐
  camada 2      │ resultados/     DADOS: a base de dados       │  entregável
                │                 (entregável do TP)           │
                └──────────────────────┬──────────────────────┘
                                       │ lê CSV
                ┌──────────────────────▼──────────────────────┐
  camada 3      │ analises/       ANÁLISE: regressões,         │  qualquer máquina,
                │                 correlações, figuras         │  segundos
                └──────────────────────┬──────────────────────┘
                                       │ escreve figuras e tabelas
                ┌──────────────────────▼──────────────────────┐
                │ relatorio/      figuras/ e tabelas/ do       │
                │                 relatório SBC                │
                └─────────────────────────────────────────────┘
```

**Por que separar.** A coleta é cara e precisa de condições controladas,
porque os tempos só são comparáveis numa máquina ociosa e sempre a mesma. A
análise é barata e muda o tempo todo: um gráfico, um modelo de regressão, uma
tabela. Com as camadas separadas:

- refazer uma figura ou trocar um modelo **não exige recoletar nada**;
- a análise pode ser escrita e testada **enquanto o B&B roda**, com os dados
  parciais;
- qualquer pessoa do grupo roda as análises no próprio computador;
- a base de dados entregue (`resultados/`) é exatamente o que as análises
  usaram.

---

## 2. Regras de dependência

| Pacote | Pode importar | Não pode importar | Lê | Escreve |
|---|---|---|---|---|
| `tsp/` | numpy, scipy, bibliotecas padrão | `experimentos`, `analises` | `TSP_Instances/` (via `carregar_instancia`) | nada |
| `experimentos/` | `tsp` | `analises` | `TSP_Instances/` | **só** `resultados/` |
| `analises/` | pandas, numpy, scipy, sklearn, matplotlib | `tsp`, `experimentos` | **só** `resultados/` | **só** `relatorio/` |

- As setas só descem: a biblioteca não sabe quem a usa, e a coleta não
  depende da análise.
- **A análise nunca roda algoritmo.** Se uma análise precisa de um dado que não
  está nos CSVs, o certo é acrescentá-lo à coleta, não calculá-lo na análise.
- [tests/test_estrutura.py](tests/test_estrutura.py) verifica as regras de
  importação a cada `make check`. Um import proibido faz o teste falhar.

---

## 3. Base: `tsp/` (algoritmos)

Código reutilizável, sem caminhos de arquivo nem CSV. Cada algoritmo aceita um
`medidor` opcional para registrar o tempo e a memória das etapas.

| Módulo | Conteúdo |
|---|---|
| `instancia.py` | leitura de `.tsp`, fecho métrico, coordenadas |
| `agm.py` | Prim O(n²) |
| `tour.py` | custo e validação de tours |
| `twice_around_the_tree.py`, `christofides.py`, `emparelhamento.py` | aproximados (Blossom próprio) |
| `heuristicas.py` | trivial, vizinho mais próximo, inserção pelo fecho convexo (MDS sem coordenadas) |
| `branch_and_bound.py` | B&B com limite de Held-Karp |
| `features.py` | features geométricas e topológicas |
| `medicao.py` | `Medidor`, `medir`, `picos_de_memoria` |

---

## 4. Camada 1: `experimentos/` (coleta)

Cada script roda algoritmos de `tsp/` e grava **dados brutos** em
`resultados/`. Todos rodam com `python -m experimentos.<script>` a partir da
raiz do repositório.

| Script | Grava em `resultados/` | Tabela do plano |
|---|---|---|
| `features.py` | `features.csv` | `instancias` |
| `aproximados_tsplib.py` | `execucoes.csv`, `subrotinas.csv` (e `_sem_gc`) | `execucoes`, `subrotinas` |
| `aproximados_aleatorios.py` | `subrotinas_aleatorias_e1.csv` | dados do E1 |
| `branch_and_bound.py` | `bnb*.csv`, `otimos_bnb.csv` | `bnb` (conjuntos `principal` e `extra`) |
| `bnb_aleatorias.py` | `bnb_aleatorias.csv` | `bnb` (conjunto `aleatorio`, regressão do E1) |
| `verificar.py` | nada (só lê) | confere se a coleta está completa (passo 5) |

Módulos de apoio, que não são scripts:

- `instancias.py`: as 12 instâncias selecionadas (`principal`), as 4 extras
  da regressão (`extra`), o rótulo `conjunto` de cada uma e o gerador
  determinístico de instâncias aleatórias (`instancia_aleatoria`);
- `ambiente.py`: processador, núcleos e versão do Python, gravados em todo
  CSV de coleta;
- `algoritmos.py`: os 5 algoritmos construtivos com interface única, usados
  como algoritmos medidos e como warm-start do B&B.

Convenções da coleta:

- **Dados brutos, sem interpretação.** Gaps contra o ótimo, regressões e
  correlações são da camada 3.
  - Exceção 1: `otimos_bnb.csv`, uma consolidação simples dos `bnb*.csv` que
    faz parte da base de dados, porque é o ótimo de referência. É gerado
    **só** com `--consolidar`, uma vez, depois de juntar os CSVs de todas as
    máquinas.
  - Exceção 2: `gap_tsplib`, que é só preliminar.
- Todo CSV do B&B tem a coluna `conjunto` (`principal`, `extra` ou
  `aleatorio`). As análises filtram por ela, nunca pelo nome do arquivo.
- O nome da instância vai sem `.tsp` (`ulysses16`, não `ulysses16.tsp`), para
  que as tabelas possam ser cruzadas pela coluna `instancia`.
- Os caminhos são calculados a partir da localização do arquivo, sem depender
  da pasta atual.
- Os scripts longos são retomáveis: o do B&B grava linha a linha e pula o
  que já está no CSV com os mesmos parâmetros, inclusive o limite de tempo.
  Um arquivo só aceita um limite de tempo, para que um teste rápido nunca se
  misture com a execução oficial.
- Todo CSV de coleta tem as colunas de ambiente (`processador`,
  `nucleos_fisicos`, `nucleos_logicos`, `python`).

---

## 5. Camada 2: `resultados/` (dados)

É a **base de dados entregável**. Só a coleta escreve aqui, e o que está aqui
vai para o repositório.

| Arquivo | Uma linha por | Origem |
|---|---|---|
| `features.csv` | instância | `experimentos.features` |
| `execucoes.csv`, `execucoes_sem_gc.csv` | algoritmo × instância | `experimentos.aproximados_tsplib` |
| `subrotinas.csv`, `subrotinas_sem_gc.csv` | algoritmo × instância × etapa | `experimentos.aproximados_tsplib` |
| `subrotinas_aleatorias_e1.csv` | algoritmo × instância aleatória × etapa | `experimentos.aproximados_aleatorios` |
| `bnb_*.csv` | instância × estratégia | `experimentos.branch_and_bound` (um arquivo por máquina); `bnb_aleatorias.csv` vem de `experimentos.bnb_aleatorias` |
| `otimos_bnb.csv` | instância | consolidação dos `bnb_*.csv` (sem as aleatórias) |

O significado das colunas está no [EXPERIMENTOS.md](EXPERIMENTOS.md): o
`bnb.csv` e o `otimos_bnb.csv` na seção 2.1; as tabelas `execucoes` e
`subrotinas` na seção 1.

---

## 6. Camada 3: `analises/` (análise)

Cada script lê CSVs de `resultados/` e grava figuras e tabelas em
`relatorio/`. Rodam em segundos e podem ser refeitos à vontade:
`python -m analises.<script>`.

| Script | Lê | Grava em `relatorio/` | Situação |
|---|---|---|---|
| `comum.py` | n/a | n/a | caminhos e `ler_resultado`, `salvar_tabela`, `salvar_figura` |
| `e1.py` | `subrotinas_aleatorias_e1.csv` | `tabelas/regressao_e1_aproximados.csv`, `figuras/e1/*.png` | **pronto** (regressão dos aproximados) |
| `gaps.py` | `execucoes.csv`, `otimos_bnb.csv` | tabela de gaps | a implementar |
| `e1.py` (B&B, GC, profiling) | `bnb_*.csv`, `subrotinas*.csv` | regressão do B&B, com × sem GC, etapa dominante | a implementar |
| `e2.py` | `bnb_*.csv`, `otimos_bnb.csv` | tabela instância × estratégia, correlações, trade-off | a implementar |
| `e3.py` | `features.csv`, gaps | modelo do gap × features (leave-one-out) | a implementar |

Convenções da análise:

- Ler sempre com `comum.ler_resultado(nome)`. Se o CSV não existir, a mensagem
  diz que é preciso rodar a coleta antes.
- Gravar sempre com `comum.salvar_tabela` e `comum.salvar_figura`, que
  escrevem em `relatorio/tabelas/` e `relatorio/figuras/`.
- A função `main` aceita as pastas como parâmetros, para que os testes rodem
  com CSVs sintéticos numa pasta temporária.

---

## 7. `relatorio/`

Figuras (`figuras/`) e tabelas (`tabelas/`) geradas pela camada 3, usadas pelo
fonte LaTeX do relatório (padrão SBC). Os arquivos gerados vão para o
repositório, para que o PDF compile sem rodar as análises. Não editar à mão:
para mudar uma figura, mude a análise e rode de novo.

---

## 8. Fluxo completo

```bash
# camada 1: coleta (máquina de referência; ver EXPERIMENTOS.md, seção 6)
python -m experimentos.features --mds
python -m experimentos.aproximados_tsplib
python -m experimentos.aproximados_aleatorios
python -m experimentos.branch_and_bound ...     # principal e extras
python -m experimentos.bnb_aleatorias            # conjunto aleatório
python -m experimentos.branch_and_bound --consolidar   # depois de juntar tudo
python -m experimentos.verificar                 # a coleta está completa?

# camada 3: análise (qualquer máquina)
python -m analises.e1
```

---

## 9. Como acrescentar algo

| Quero… | Onde |
|---|---|
| um novo algoritmo ou heurística | `tsp/` (com testes), e incluir em `experimentos/algoritmos.py` |
| medir algo novo | um script em `experimentos/` que grave em `resultados/`; documentar as colunas no EXPERIMENTOS.md |
| uma nova figura, tabela ou estatística | um script em `analises/` que leia de `resultados/`; nunca rodar algoritmo |
| testar | `tests/test_<módulo>.py`; análises com CSVs sintéticos em `tmp_path` |
