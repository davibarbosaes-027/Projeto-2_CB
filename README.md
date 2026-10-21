# Z.Data — Indicadores contábeis e entrada em distress financeiro

Projeto do Insper Data, Ciclo Básico 2026.2.

> Rascunho. As seções marcadas como *(a preencher)* dependem das próximas fases.

## Problema e pergunta

Empresas que entram em distress financeiro, ou seja, que deixam de gerar resultado operacional suficiente para pagar os juros da dívida, impõem perdas a credores, acionistas e funcionários. Antecipar esses casos é o objetivo de modelos de risco de crédito como o Z-score de Altman.

**Pergunta:** indicadores contábeis (alavancagem, rentabilidade, geração de caixa, liquidez, exposição cambial, tamanho e setor) estão associados à **entrada** de companhias abertas brasileiras não financeiras em distress financeiro nos dois anos seguintes?

O estudo é de **associação e previsão, não de causalidade**.

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

## Definição do alvo (Fase 2: `notebooks/02_base.ipynb`)

- `cobertura_t = EBIT_t / despesa financeira_t`, usando o ano completo (4º trimestre). Se a despesa for menor ou igual a zero (contaminação cambial), a cobertura é NaN.
- `distress_t = 1` se `cobertura_t < 1`.
- `Y_t = 1` se a empresa está em distress em **t+1 e t+2**; `Y_t = 0` se os dois anos são classificáveis e não estão ambos em distress.
- Amostra em risco: empresas **fora** de distress em t (`cobertura_t ≥ 1`), com t de 2010 a 2023.
- Resultado: **1.708 empresa-anos, 246 empresas e 197 eventos (11,5%)**.
- Divisão temporal sem embaralhar os anos e com 2 anos de intervalo: treino t = 2010–2018 (957 obs., 12,7% de eventos), validação t = 2019–2020 (276 obs., 7,6%) e teste t = 2021–2023 (475 obs., 11,4%).

## Variáveis explicativas

| Variável | Definição | Hipótese |
|---|---|---|
| alavancagem | dívida bruta / ativo total | H1 (+) |
| divida_cp_share | empréstimos CP / (CP + LP) | H1 (+) |
| roa | lucro líquido 12m / ativo total | H2 (−) |
| liquidez | caixa / passivo circulante | H3 (−) |
| fco_divida | caixa operacional 12m / dívida bruta | H4 (−) |
| log_ativo, setor, ano | controles | |

**H5 (exposição cambial) foi abandonada**, porque o Economatica só trouxe a dívida em moeda estrangeira para 2010. O setor é **provisório**: um agrupamento em macrossetores do setor da CVM, que será trocado pelo Subsetor Bovespa quando a dupla exportá-lo do Economatica.

Nenhuma explicativa usa EBIT ou EBITDA, porque eles estão no alvo. A baseline é o Z''-score de Altman para mercados emergentes.

## Decisões de limpeza

Ver `notebooks/01_inspecao.ipynb` (Fase 1). Resumo:
- só o 4º trimestre de cada ano: resolve as variáveis acumuladas no ano (receita e despesa financeira);
- `passivo_mais_pl` é descartada porque é igual ao ativo total;
- `ResLuc` é Reservas de Lucro. O valor negativo é o saldo de reservas menos prejuízos acumulados (plano padronizado do Economatica), e o sinal é mantido;
- 8 financeiras que escaparam do Screening foram excluídas (B1003, B3SA3, BRBI11, CXSE3, HBTS5, ITSA4, QUAL3 e WIZC3);
- os anos anteriores ao registro na CVM (demonstrações do prospecto de IPO) são mantidos, com a flag `pre_registro`;
- divisão por zero vira NaN; empresas sem dívida recebem a flag `sem_divida`;
- winsorização nos percentis 1% e 99%, com os limites calculados só no treino.

## Limitação: viés de sobrevivência

O *Screening* trouxe só as empresas **listadas hoje**. Das 469 companhias não financeiras que cancelaram o registro na CVM entre 2010 e 2026, **nenhuma** está na base. Como as empresas que quebram ou fecham capital são justamente candidatas a distress, a taxa de eventos tende a ser **subestimada**, principalmente nos primeiros anos, e a amostra de treino tende a ser mais "saudável" que a realidade. Medição (`notebooks/03_exploratoria.ipynb`):
- a base cobre **47%** das companhias não financeiras com ações em bolsa em 2010 e **73%** em 2025;
- nenhuma das 469 canceladas desde 2010 está na base, e pelo menos 31 delas eram listadas e estavam em crise (recuperação judicial, falência ou cancelamento de ofício);
- consequência: a taxa de eventos está subestimada e as associações tendem a ser conservadoras. As probabilidades devem ser lidas como risco relativo.

## Como rodar

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_inspecao.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_base.ipynb        # gera data/processed/
jupyter nbconvert --to notebook --execute --inplace 03_exploratoria.ipynb  # gera figures/
```

Estrutura:

```
data/raw/        dados brutos, nunca editados
data/processed/  bases geradas pelo código (não versionadas)
notebooks/       01_inspecao, 02_base, 03_exploratoria
src/             config.py (constantes), dados.py (leitura), verificacoes.py, base.py (Fase 2), eda.py (Fase 3), style.py (gráficos), fonts/
figures/         PNGs para os slides
```

## Resultados *(a preencher)*

## Limitações *(a preencher)*
