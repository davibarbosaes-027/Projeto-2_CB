# Z.Data — Indicadores contábeis e entrada em distress financeiro

Projeto final do Ciclo Básico 2026.2 do Insper Data (dupla: Diego dos Santos Mitsunaga e Davi Barbosa do Espirito Santo).

## Resumo

**Pergunta:** indicadores contábeis estão associados à **entrada** de companhias abertas brasileiras não financeiras em distress financeiro (EBITDA / despesa financeira < 1) nos dois anos seguintes?

**Resposta curta (associação, não causalidade):**
- **Sim, no sentido esperado.** Controlando pelas demais variáveis, mais dívida (alavancagem e dívida de curto prazo) e menos rentabilidade (ROA) e menos geração de caixa (FCO/dívida) estão associadas a mais entrada em distress; a liquidez não acrescenta nada. A evidência é **moderada** (p entre 0,04 e 0,09, intervalos largos).
- **Regressão e machine learning concordam no essencial:** FCO/dívida e dívida de curto prazo são as variáveis mais importantes nos três modelos.
- **Como previsor, os indicadores ordenam o risco, mas não superam o Z'' de Altman recalibrado.** No teste (2021–2023, 31 eventos) a logística tem AUC-PR de 0,13 (a de um modelo sem informação é 0,057), empatada com o Z'' (0,14). Com tão poucos eventos, o teste não distingue diferenças menores que ~0,07.
- **Limitações importantes:** poucos eventos, viés de sobrevivência (nenhuma das 469 companhias que cancelaram o registro está na base), alvo baseado na despesa financeira bruta do Economatica, setor provisório e sem controle de tempo.

## Problema e pergunta

Empresas que entram em distress financeiro, ou seja, que deixam de gerar resultado operacional suficiente para pagar os juros da dívida, impõem perdas a credores, acionistas e funcionários. Antecipar esses casos é o objetivo de modelos de risco de crédito como o Z-score de Altman.

**Hipóteses (escritas antes de modelar):**

| | Indicador | Efeito esperado na chance de distress | Por quê |
|---|---|---|---|
| H1 | Alavancagem (dívida bruta / ativo) e dívida de curto prazo (CP / total) | aumenta | mais dívida, mais juros a pagar com o mesmo lucro |
| H2 | Rentabilidade (ROA) | diminui | mais lucro sobra para pagar os juros |
| H3 | Liquidez (caixa / passivo circulante) | diminui | caixa dá fôlego para pagar o curto prazo |
| H4 | Geração de caixa (FCO / dívida) | diminui | caixa operacional paga juros e amortiza dívida |

Tamanho (log do ativo) e setor entram como controles. **H5 (exposição cambial) foi abandonada**, porque o Economatica só trouxe a dívida em moeda estrangeira para 2010. O estudo é de **associação e previsão, não de causalidade**.

## Dados

### Economatica (licenciado: não redistribuir)

A base do Economatica é **licenciada e não pode ser redistribuída**. Por isso os CSVs não estão no repositório: `data/raw/economatica/` está no `.gitignore`. Quem tiver acesso ao Economatica (o Insper tem) consegue gerar os arquivos assim:

1. **Screening** (lista de empresas):
   - Universo: ações da B3 (bolsa `XBSP`), companhias brasileiras, **um ticker por empresa** (o de maior liquidez).
   - Filtro: **excluir bancos, seguradoras e demais financeiras** pelo setor.
   - Atenção: o Screening padrão traz só ativos **ativos hoje**. Se possível, marque a opção de **incluir ativos cancelados**. A nossa extração não incluiu, e isso causa o viés de sobrevivência descrito abaixo.
2. **Matrixx** (painel):
   - Linhas: empresa × data, periodicidade **trimestral**, de **1T2010 a 4T2026**.
   - Configurações de todas as colunas: **moeda original**, **em milhares**, **consolidado: sim** (com fallback para individual, o `*` do Economatica).
   - Colunas (entre parênteses, o nome que aparece no cabeçalho exportado e a janela):

     | grupo | colunas |
     |---|---|
     | Balanço (na data) | Ativo Total (`Ativo Tot`), Ativo Circulante (`AtvCir`), Passivo Circulante (`PasCir`), Patrimônio Líquido (`Patrim Liq`), Passivo + PL (`Pas+PL`), Caixa e Equivalentes (`CaixaEEqCx`), Empréstimos e Financiamentos CP (`TotEmFiCP`) e LP (`TotEmFiLP`), Dívida Total Bruta (`Div Tt Bruta`), Dívida Total Líquida (`Div Tt Lq`), Reservas de Lucro (`ResLuc`), Dívida em Moeda Estrangeira (`Div Moed Estr`) |
     | Resultado, 12 meses | EBIT (`EBIT`), EBITDA (`EBITDA`), Lucro Líquido (`Lucro Liquido`), Caixa Operacional (`CxOper`, sai em R$ e não em milhares) |
     | Resultado, no exercício (acumulado no ano) | Receita (`Receita`), Despesa Financeira (`DesFin`), EBITDA/Despesa Financeira (`Ebitda/Fin`) |
     | Outros | Liquidez Corrente (`LiqCor`), Valor de Mercado da empresa (`Valor Mercado`), Juros Pagos (`JurosPagos`), Capex (`Capex (exc inv)`) |

3. **Exportar em CSV**, mantendo as colunas `Ativo` e `Data` em todos os arquivos. O Economatica limita o nº de colunas por exportação, por isso a extração foi feita em 3 arquivos: `Economatica-<id>.csv`, `extra.csv` e `juros_pagaos_&_capex.csv`. Salve-os em `data/raw/economatica/` e ajuste os nomes em `src/config.py` (`ARQ_ECONOMATICA`), se forem outros.

O código casa as colunas pelo prefixo do cabeçalho (antes do primeiro `|`, ver `cfg.RENOMEAR`). Por isso a ordem das colunas e o número de arquivos podem variar.

Cuidados que o código já trata:
- `"-"` significa sem dado e vira NaN, nunca zero;
- `CxOper` e `JurosPagos` vêm em R$, enquanto o resto vem em milhares;
- `ResLuc` é **Reservas de Lucro**, não resultado líquido;
- `JurosPagos` e `Capex` vieram vazios e não são usados.

### CVM (Dados Abertos, público)

- Cadastro de companhias abertas: <https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv> → `data/raw/cvm/`
- Formulário Cadastral (FCA), um zip por ano de 2010 a 2026: `https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_AAAA.zip` → `data/raw/cvm/fca/fca_AAAA.zip`. Usamos a tabela `valor_mobiliario`, que liga o código de negociação (ticker) ao CNPJ, porque a base do Economatica não traz CNPJ.

## Definição do alvo (notebook 04)

O alvo foi escolhido **antes de olhar o desempenho de qualquer modelo**, por três critérios (prevalência próxima da literatura, sinalizar os eventos reais de crédito conhecidos e ter eventos suficientes no treino). Detalhes e comparação das quatro definições (A a D) em `notebooks/04_alvo.ipynb`.

- `cobertura_t = EBITDA_t / despesa financeira_t`, com o ano completo (4º trimestre). Se a despesa financeira for menor ou igual a zero (contaminação cambial), a cobertura não é classificável.
- `distress_t = 1` se `cobertura_t < 1`.
- `Y_t = 1` se a empresa está em distress em **t+1 e t+2** (definição **B**; `cfg.ALVO_MODELO`).
- **Amostra em risco:** empresas **fora** de distress em t, com `Y` observado, sem os anos anteriores ao registro na CVM e com t de 2010 a 2023.
- Resultado: **1.845 empresa-anos, 243 empresas e 124 eventos (6,7%)**.
- **Divisão temporal**, sem embaralhar os anos e com intervalo de 2 anos (o rótulo de t usa t+1 e t+2):

| | t | empresa-anos | eventos | taxa |
|---|---|---|---|---|
| treino | 2010–2018 | 1.013 | 82 | 8,1% |
| validação | 2019–2020 | 286 | 11 | 3,8% |
| teste | 2021–2023 | 546 | 31 | 5,7% |

- O EBIT (definição A) fica como teste de robustez: com EBIT, cerca de 42% das empresas ficam abaixo de 1 em um ano, contra 29% com EBITDA, que fica mais perto da literatura.

## Variáveis explicativas

| Variável | Definição | Hipótese |
|---|---|---|
| alavancagem | dívida bruta / ativo total | H1 (+) |
| divida_cp_share | empréstimos CP / (CP + LP) | H1 (+) |
| roa | lucro líquido 12m / ativo total | H2 (−) |
| liquidez | caixa / passivo circulante | H3 (−) |
| fco_divida | caixa operacional 12m / dívida bruta | H4 (−) |
| log_ativo, setor | controles | |
| sem_divida | flag para empresas sem dívida | |

Nenhuma explicativa usa EBIT ou EBITDA, porque eles estão no alvo. A baseline é o Z''-score de Altman para mercados emergentes. O setor é **provisório**: um agrupamento em macrossetores do setor da CVM, que será trocado pelo Subsetor Bovespa quando a dupla exportá-lo do Economatica.

## Decisões de limpeza

Ver `notebooks/01_inspecao.ipynb` e `notebooks/02_base.ipynb`. Resumo:
- só o 4º trimestre de cada ano: resolve as variáveis acumuladas no ano (receita e despesa financeira);
- `passivo_mais_pl` é descartada porque é igual ao ativo total;
- `ResLuc` é Reservas de Lucro. O valor negativo é o saldo de reservas menos prejuízos acumulados (plano padronizado do Economatica), e o sinal é mantido;
- 8 financeiras que escaparam do Screening foram excluídas (B1003, B3SA3, BRBI11, CXSE3, HBTS5, ITSA4, QUAL3 e WIZC3);
- os anos anteriores ao registro na CVM (demonstrações do prospecto de IPO) são mantidos na base, com a flag `pre_registro`, mas ficam fora da amostra em risco;
- divisão por zero vira NaN; empresas sem dívida recebem a flag `sem_divida`;
- winsorização nos percentis 1% e 99%, com os limites calculados **só no treino** (dentro do `Pipeline` na modelagem, para não vazar).

## Principais achados da exploração (notebook 03, amostra do alvo B, treino)

Taxa de entrada em distress por quintil de cada variável (do Q1 ao Q5): **FCO/dívida** 16,8% → 2,1% (o padrão mais forte, monotônico); **liquidez** 12,8% → 3,4%; **ROA** 12,3% → 4,4%; **alavancagem** 5,9% → 10,3% (com platô nos dois últimos quintis); **dívida de curto prazo** 6,6% → 12,3%; **tamanho** 13,3% → 4,9% (controle). O Z'' ordena pouco o risco (9,9%, 6,9%, 7,9%, 9,9% e 5,9%). A taxa de eventos varia de 0% (2012) a 19,5% (2014), então o ciclo econômico pesa muito. Não há multicolinearidade (maior |r| de Spearman = 0,54; VIF < 1,6), então nenhuma variável sai por redundância. Decisão de modelagem: logística com regularização L2 (poucos eventos), avaliação por AUC-PR e validação temporal.

## Métodos (notebook 05)

- **Pipelines sem vazamento** (scikit-learn): winsorização, regra do sem-dívida, imputação pela mediana, padronização e dummies de macrossetor são ajustadas **só no treino de cada dobra**. Um teste automático embaralha os rótulos do teste e confere que o ajuste não muda.
- **Hiperparâmetros por validação temporal expandida dentro do treino** (valida em 2015–2018, treina com o passado até 2 anos antes); o teste (2021–2023) é aberto **uma vez**, no fim. O modelo final é ajustado só com o treino.
- **Regressão de referência e inferência:** logística sem penalização com **erro-padrão robusto agrupado por empresa** (o painel repete empresas) e **modelo linear de probabilidade** (efeito em pontos percentuais), com VIF e checagem de resíduos. "Confirma" = sinal esperado e p < 0,10.
- **Modelos comparados:** Z'' recalibrado (baseline), logística L2 (com e sem pesos de classe; a principal usa pesos), logística com splines, gradient boosting e random forest (desafiantes, hiperparâmetros fixos).
- **Comparação regressão × ML:** importância por permutação (só com dados de treino).
- **Métricas:** AUC-PR (principal; um modelo sem informação tem AUC-PR igual à taxa de eventos), AUC-ROC, Brier (contra a previsão ingênua da taxa do treino), recall no top 20% e precisão/recall no limiar (F2). Intervalos de 90% por **bootstrap por empresa**, e diferenças contra o Z'' na mesma reamostragem.

## Resultados

**1. Hipóteses (logística sem penalização no treino, razão de chances por +1 desvio-padrão, IC de 90%):**

| Variável | Razão de chances | p | Efeito linear (p.p.) | Veredito |
|---|---|---|---|---|
| Alavancagem (H1 +) | 1,39 (1,04–1,87) | 0,064 | +3,0 | confirma |
| Dívida de curto prazo (H1 +) | 1,36 (1,01–1,82) | 0,086 | +2,2 | confirma |
| ROA (H2 −) | 0,72 (0,53–0,98) | 0,083 | −2,6 | confirma |
| Liquidez (H3 −) | 0,85 (0,61–1,18) | 0,418 | −0,8 | não confirma |
| FCO / dívida (H4 −) | 0,29 (0,10–0,79) | 0,043 | −0,6 (p = 0,34) | confirma na logística, **não** no modelo linear |
| Tamanho (controle) | 0,75 (0,57–1,00) | 0,097 | −2,5 | menos risco |

Nenhum setor é significativo. A flag `sem_divida` (razão de chances 3,8, IC 1,3–11,5) vem de 5 eventos e não ajuda fora da amostra: não interpretar. Resíduos e VIF (máximo 1,99) não mostram problema sério; o modelo linear tem R² de 4,6% e 7,8% das probabilidades ajustadas negativas, por isso a logística é a principal.

**2. Regressão e machine learning:** a importância por permutação põe **FCO/dívida em 1º e dívida de curto prazo em 2º nos três modelos** (logística, gradient boosting e random forest); ROA e liquidez trocam de posição entre os modelos.

**3. Previsão no teste (2021–2023, 546 empresa-anos, 31 eventos; taxa 5,7%):**

| Modelo | AUC-PR | AUC-ROC | Recall no top 20% |
|---|---|---|---|
| Sem informação | 0,057 | 0,500 | 20% |
| Z'' recalibrado (baseline) | 0,138 | 0,633 | 42% |
| **Logística L2 (principal)** | **0,134** | **0,673** | **32%** |
| Logística sem pesos | 0,135 | 0,680 | 35% |
| Logística com splines | 0,116 | 0,649 | 42% |
| Gradient boosting | 0,156 | 0,742 | 55% |
| Random forest | 0,148 | 0,755 | 58% |

- **Logística contra Z'':** diferença de AUC-PR de −0,007 (IC 90% de −0,10 a +0,07; P(logística > Z'') = 0,45): **empate técnico**. Os intervalos de 90% da AUC-PR de todos os modelos vão de ~0,08 a ~0,25.
- **Árvores:** têm AUC-ROC maior que o Z'' (+0,11 e +0,12, ICs que excluem zero por pouco), mas não na AUC-PR; com 10 comparações, é uma pista de não linearidade, não uma conclusão.
- **Uso prático:** o limiar da logística marca 74 de 546 empresas e capta 8 dos 31 distress (recall de 26%), com 66 alarmes falsos (precisão de 11%, o dobro da taxa-base). O limiar do Z'' marca 98% das empresas (foi escolhido em anos de taxa mais alta). Uma regra "os 20% mais arriscados" é mais robusta.
- **Probabilidades:** os pesos de classe inflam a probabilidade da logística principal (16% a 63% previstos contra 2% a 9% observados); use-a como **ranking**. Contra a previsão ingênua (Brier 0,0541), o Z'' ganha 0,7% e a logística sem pesos 2,7%: em probabilidade absoluta praticamente empatam.
- **Robustez (alvo A com EBIT, sem empresas sem dívida, sem controle de setor):** a conclusão se mantém (nem melhor nem pior que o Z''); com o alvo A o resultado muda de sinal entre as métricas, então depende da definição do alvo.
- **Sensibilidade a C:** a curva de validação é plana a partir de C ≈ 0,3, e a regra "maior AUC-PR" escolhe C = 1.000 (quase sem penalização). Com C = 0,3 as métricas do teste mudam de forma comparável ao ruído (AUC-ROC da logística 0,61 contra 0,67).

## Limitações

- **Poucos eventos** (82, 11 e 31): intervalos largos; o teste não distingue diferenças de ~0,07 de AUC-PR. Cerca de 6 eventos por parâmetro, abaixo dos 10 recomendados (por isso a regularização L2).
- **Viés de sobrevivência.** O *Screening* trouxe só as empresas **listadas hoje**. Das 469 companhias não financeiras que cancelaram o registro na CVM entre 2010 e 2026, **nenhuma** está na base (a cobertura é de 47% em 2010 e 73% em 2025), e pelo menos 31 delas eram listadas e estavam em crise. A taxa de eventos tende a ser **subestimada** e as associações, **conservadoras**; as probabilidades devem ser lidas como risco relativo.
- **Alvo:** baseado em EBITDA e na **despesa financeira bruta** do Economatica (o câmbio contamina a despesa; despesa ≤ 0 fica fora); `JurosPagos` veio vazio. Com o EBIT, o resultado muda.
- **Validação com eventos reais** (RJ, default): dos 8 casos da lista da dupla, só 3 puderam ser avaliados (Americanas, Light e Azul), pois os demais não estão na base ou não têm despesa financeira; os 3 estavam em distress em t−1.
- **Setor provisório e sem controle de tempo** (juros, Selic): a taxa de eventos varia entre períodos e desloca o limiar.
- **Variáveis que ficaram de fora:** governança, cronograma da dívida, exposição cambial (H5), preço de mercado.
- **A flag `sem_divida`** foi estimada com 5 eventos e não se sustenta fora da amostra.
- **Hiperparâmetros** das árvores fixos (sem tuning), de propósito.
- Os resultados são **associações**, não efeitos causais.

## Como rodar

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_inspecao.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_base.ipynb          # gera data/processed/base_empresa_ano.parquet
jupyter nbconvert --to notebook --execute --inplace 03_exploratoria.ipynb  # EDA (figuras 01 a 16)
jupyter nbconvert --to notebook --execute --inplace 04_alvo.ipynb          # escolha do alvo
jupyter nbconvert --to notebook --execute --inplace 05_modelagem.ipynb     # modelos e resultados (figuras 17 a 23)
cd .. && python -m pytest tests -q                                         # 32 testes, com painel sintético
```

Os notebooks 01 e 02 precisam dos CSVs do Economatica (licenciados; veja acima). A base derivada `data/processed/base_empresa_ano.parquet` também não é versionada: peça à dupla ou gere pelo notebook 02. Os testes usam um painel **sintético** (`tests/sintetico.py`), só para validar o código, nunca para resultados.

Estrutura:

```
data/raw/        dados brutos, nunca editados
data/processed/  bases geradas pelo código (não versionadas)
notebooks/       01_inspecao, 02_base, 03_exploratoria, 04_alvo, 05_modelagem
src/             config.py (constantes), dados.py, verificacoes.py, base.py (Fase 2), alvo.py (alvos A a D),
                 eda.py (Fase 3), modelagem.py e avaliacao.py (Fase 4), style.py (gráficos), fonts/
tests/           testes automáticos (pytest) com painel sintético
figures/         PNGs usados nos slides (fig01 a fig23)
```