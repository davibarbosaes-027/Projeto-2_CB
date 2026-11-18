# Dicionário de variáveis — base_empresa_ano.parquet

Uma linha por empresa × ano (4º trimestre). Valores em R$ mil, salvo indicação.

| coluna | tipo | % NaN | descrição |
|---|---|---|---|
| `ticker` | str | 0,0% | Código de negociação na B3 (sem o sufixo <XBSP> do Economatica; nas empresas da DFP, o código do FCA) |
| `ano` | int64 | 0,0% | Ano t (dados do 4º trimestre = ano completo) |
| `desp_financeira` | float64 | 1,9% | Despesa financeira do ano (R$ mil, positiva = despesa) |
| `ativo_total` | float64 | 0,0% | Ativo total (R$ mil, na data) |
| `divida_bruta` | float64 | 0,1% | Dívida total bruta = empréstimos CP + LP (R$ mil) |
| `divida_liquida` | float64 | 0,1% | Dívida total líquida (Economatica, R$ mil) |
| `patrimonio_liquido` | float64 | 0,0% | Patrimônio líquido (R$ mil) |
| `caixa` | float64 | 0,1% | Caixa e equivalentes (R$ mil) |
| `emprestimos_cp` | float64 | 0,1% | Empréstimos e financiamentos de curto prazo (R$ mil) |
| `emprestimos_lp` | float64 | 0,1% | Empréstimos e financiamentos de longo prazo (R$ mil) |
| `reservas_lucro` | float64 | 0,0% | Reservas de lucro (R$ mil). Negativo = prejuízos acumulados maiores que as reservas (plano padronizado do Economatica) |
| `ativo_circulante` | float64 | 0,0% | Ativo circulante (R$ mil) |
| `passivo_circulante` | float64 | 0,0% | Passivo circulante (R$ mil) |
| `receita` | float64 | 0,0% | Receita líquida do ano (R$ mil) |
| `valor_mercado` | float64 | 45,1% | Valor de mercado da empresa (R$ mil). Não usado nas explicativas |
| `ebit` | float64 | 0,1% | EBIT 12 meses (R$ mil). SÓ usado no alvo e no Z'' |
| `ebitda` | float64 | 3,0% | EBITDA 12 meses (R$ mil). Não usado |
| `lucro_liquido_12m` | float64 | 0,1% | Lucro líquido 12 meses (R$ mil) |
| `caixa_operacional_12m` | float64 | 0,1% | Fluxo de caixa operacional 12 meses (R$ mil; exportado em R$ e convertido) |
| `fonte` | str | 0,0% | Economatica (empresas listadas hoje) ou CVM (DFP) (empresas fora do Screening, quase todas canceladas) |
| `CNPJ_CIA` | str | 0,3% | CNPJ da companhia (ligado pelo FCA da CVM; ver como_casou) |
| `como_casou` | str | 0,0% | Como o ticker foi ligado ao CNPJ: exato, raiz (4 letras), manual, manual_conferir, sem_match |
| `DENOM_SOCIAL` | str | 0,4% | Razão social no cadastro da CVM |
| `SIT` | str | 0,4% | Situação atual do registro na CVM |
| `DT_REG_PRIMEIRO` | datetime64[us] | 0,4% | Primeira data de registro na CVM |
| `DT_CANCEL` | datetime64[us] | 82,9% | Data de cancelamento do registro (se houver) |
| `MOTIVO_CANCEL` | str | 82,9% | Motivo do cancelamento (CVM) |
| `SETOR_ATIV` | str | 0,4% | Setor de atividade da CVM (original) |
| `setor` | str | 0,0% | Macrossetor PROVISÓRIO (agrupamento do SETOR_ATIV; será trocado pelo Subsetor Bovespa) |
| `ano_registro` | float64 | 0,4% | Ano do primeiro registro na CVM |
| `idade_registro` | float64 | 0,4% | t − ano de registro (negativo = demonstração anterior ao registro, ex.: prospecto de IPO) |
| `pre_registro` | bool | 0,0% | True se t é anterior ao ano de registro na CVM |
| `ano_evento_credito` | float64 | 88,3% | Ano do 1º evento de crédito (RJ, RE, falência, liquidação ou saída da bolsa em crise); NaN se nenhum |
| `tipo_evento_credito` | str | 88,3% | Tipo do evento de crédito |
| `fonte_evento` | str | 88,3% | De onde veio o evento: CVM (cadastro) ou lista manual (data/raw/eventos) |
| `cobertura` | float64 | 5,0% | EBIT / despesa financeira em t; NaN se despesa ≤ 0 ou ausente |
| `distress` | float64 | 5,0% | 1 se cobertura < 1; 0 se ≥ 1; NaN se não classificável |
| `distress_t1` | float64 | 13,5% | distress em t+1 (procurado por ano) |
| `presente_t1` | bool | 0,0% | Empresa tem balanço em t+1 |
| `distress_t2` | float64 | 21,8% | distress em t+2 |
| `presente_t2` | bool | 0,0% | Empresa tem balanço em t+2 |
| `Y` | float64 | 21,5% | ALVO: 1 se distress em t+1 e t+2 OU evento de crédito em t+1..t+2; 0 se ambos classificáveis e nenhum dos dois; NaN caso contrário |
| `origem_Y` | str | 69,8% | Por que Y = 1: cobertura (2 anos < 1), evento (de crédito em t+1..t+2) ou os dois |
| `em_risco` | bool | 0,0% | True se está na amostra de modelagem: cobertura_t ≥ 1, Y observado, sem evento de crédito até t e t ≤ 2023 |
| `alavancagem` | float64 | 0,2% | H1 (+): dívida bruta / ativo total |
| `divida_cp_share` | float64 | 11,1% | H1 (+): empréstimos CP / (CP + LP); NaN se sem dívida |
| `roa` | float64 | 0,2% | H2 (−): lucro líquido 12m / ativo total |
| `liquidez` | float64 | 0,7% | H3 (−): caixa / passivo circulante |
| `sem_divida` | bool | 0,0% | True se dívida bruta = 0 (fco_divida e divida_cp_share ficam NaN) |
| `fco_divida` | float64 | 11,2% | H4 (−): caixa operacional 12m / dívida bruta; NaN se dívida = 0 |
| `log_ativo` | float64 | 0,1% | Controle de tamanho: log(ativo total em R$ mil) |
| `z_x1` | float64 | 0,1% | Z'': (AC − PC) / AT |
| `z_x2` | float64 | 0,1% | Z'': reservas de lucro / AT |
| `z_x3` | float64 | 0,2% | Z'': EBIT / AT |
| `z_x4` | float64 | 0,5% | Z'': PL / (AT − PL); NaN se passivo ≤ 0 |
| `z_score` | float64 | 0,6% | Z''-score (Altman, emergentes) = 3,25 + 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4; menor = mais risco |
| `split` | object | 12,7% | treino (t 2010–2018), validacao (2019–2020), teste (2021–2023) |
| `alavancagem_w` | float64 | 0,2% | alavancagem winsorizada nos percentis 1% e 99% do treino |
| `divida_cp_share_w` | float64 | 11,1% | divida_cp_share winsorizada nos percentis 1% e 99% do treino |
| `roa_w` | float64 | 0,2% | roa winsorizada nos percentis 1% e 99% do treino |
| `liquidez_w` | float64 | 0,7% | liquidez winsorizada nos percentis 1% e 99% do treino |
| `fco_divida_w` | float64 | 11,2% | fco_divida winsorizada nos percentis 1% e 99% do treino |
| `log_ativo_w` | float64 | 0,1% | log_ativo winsorizada nos percentis 1% e 99% do treino |
