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
   - Atenção: o Screening padrão traz só ativos **ativos hoje**. A nossa extração não incluiu os cancelados; as empresas que faltam entram pela DFP da CVM (ver "Viés de sobrevivência", abaixo).
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
- Formulário Cadastral (FCA), um zip por ano de 2010 a 2026: `https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_AAAA.zip` → `data/raw/cvm/fca/fca_AAAA.zip`. Usamos a tabela `valor_mobiliario`, que liga o código de negociação (ticker) ao CNPJ, porque a base do Economatica não traz CNPJ, e que diz quais companhias tinham ações em bolsa.
- Demonstrações Financeiras Padronizadas (DFP), um zip por ano de 2010 a 2026: `https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_AAAA.zip` → `data/raw/cvm/dfp/dfp_AAAA.zip`. **O código baixa sozinho** na primeira execução (`src/cvm_dfp.py`; ~170 MB, fora do git). Traz o balanço anual de todas as companhias abertas, **inclusive as canceladas**, e é a fonte das empresas que o Screening do Economatica não trouxe.

## Definição do alvo (Fase 2: `notebooks/02_base.ipynb`)

- `cobertura_t = EBIT_t / despesa financeira_t`, usando o ano completo (4º trimestre). Se a despesa for menor ou igual a zero (contaminação cambial), a cobertura é NaN.
- `distress_t = 1` se `cobertura_t < 1`.
- `Y_t = 1` se a empresa está em distress em **t+1 e t+2**, **ou** se tem um **evento de crédito** em t+1 ou t+2; `Y_t = 0` se os dois anos são classificáveis e nenhuma das duas coisas acontece.
- **Evento de crédito** (`cfg.EVENTO_CREDITO_NO_ALVO`): recuperação judicial ou extrajudicial, falência, liquidação ou saída da bolsa em crise (cancelamento de ofício pela CVM, que cancela quem parou de publicar). Fontes: cadastro da CVM e `data/raw/eventos/eventos_credito.csv`. Existe porque a empresa que quebra costuma **parar de publicar balanço**: sem essa regra, o `Y` dela ficaria NaN e o caso mais importante sairia da amostra. Cancelamento voluntário (OPA) e incorporação não são eventos.
- Amostra em risco: empresas **fora** de distress em t (`cobertura_t ≥ 1`) e sem evento de crédito até t, com t de 2010 a 2023.
- Resultado (alvo com EBIT): **2.354 empresa-anos, 377 empresas e 301 eventos (12,8%)**, dos quais 103 em empresas que só existem na DFP. 60 empresas têm evento de crédito.
- Divisão temporal sem embaralhar os anos e com 2 anos de intervalo: treino t = 2010–2018 (1.391 obs., 14,2% de eventos), validação t = 2019–2020 (369 obs., 9,2%) e teste t = 2021–2023 (594 obs., 11,8%).

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

## Viés de sobrevivência: problema e correção

**O problema.** O *Screening* do Economatica trouxe só as empresas **listadas hoje**. Nenhuma das companhias não financeiras que cancelaram o registro entre 2010 e 2026 estava na base, e a cobertura era de só 47% das listadas em 2010 (73% em 2025). As empresas que quebram e saem da bolsa são justamente as que o modelo precisa ver: sem elas, ele aprende a separar empresas boas de empresas ruins **que se recuperaram**.

**A correção** (`src/cvm_dfp.py`, notebook 02, seções 3b e 4a):
1. As companhias não financeiras com ações em bolsa que não estão no Economatica entram pela **DFP da CVM**: **223 empresas e 1.993 empresas-anos**, quase todas canceladas.
2. **Validação:** para as 289 empresas que estão nas duas fontes, a DFP reproduz o Economatica a menos de 1% em ~98% das empresas-ano (balanço, receita, EBIT, despesa financeira, FCO; lucro e EBITDA ~90%), e a classificação de distress concorda em **mais de 99%**.
3. O alvo ganhou o **evento de crédito** (ver acima), para que a empresa que para de publicar balanço em crise conte como evento.

**Resultado:** a base cobre **92–96%** das companhias não financeiras com ações em bolsa em todos os anos. Nas empresas que faltavam, a taxa de eventos é **15,9%**, contra 11,6% nas do Economatica, o que confirma que o viés deixava a amostra mais "saudável" que a realidade.

**O que ainda falta:**
- 12 companhias canceladas em 2010–2011 não publicaram DFP no período e continuam fora;
- o cadastro da CVM só traz a situação **atual** do emissor: quem entrou em RJ e já saiu (ex.: PDG) não aparece como evento, embora o distress dela continue captado pela cobertura;
- as empresas da DFP não têm valor de mercado (não é usado nas explicativas).

## Como rodar

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_inspecao.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_base.ipynb        # baixa a DFP na 1ª vez e gera data/processed/
jupyter nbconvert --to notebook --execute --inplace 03_exploratoria.ipynb  # gera figures/
jupyter nbconvert --to notebook --execute --inplace 04_alvo.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_modelagem.ipynb
```

Estrutura:

```
data/raw/        dados brutos, nunca editados (economatica/ e cvm/dfp/ fora do git)
data/processed/  bases geradas pelo código (não versionadas)
notebooks/       01_inspecao, 02_base, 03_exploratoria, 04_alvo, 05_modelagem
src/             config.py (constantes), dados.py (leitura), cvm_dfp.py (DFP da CVM), verificacoes.py, base.py (Fase 2),
                 eda.py (Fase 3), alvo.py, modelagem.py, avaliacao.py, style.py (gráficos), fonts/
figures/         PNGs para os slides
```

## Resultados *(a preencher)*

## Limitações *(a preencher)*
