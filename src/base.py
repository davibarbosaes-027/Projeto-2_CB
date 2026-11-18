"""Fase 2: base empresa-ano, alvo, explicativas, Z''-score, divisão temporal e winsorização."""
import numpy as np
import pandas as pd

from . import config as cfg, dados


def _div(a, b):
    """Divisão que devolve NaN quando o denominador é zero ou ausente (nunca inf)."""
    return (a / b.where(b != 0)).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- 1. empresa-ano
def empresa_ano(df_trim):
    """Fica só com o 4º trimestre (no 4T, 'no exercício' = ano completo = '12 meses').
    Linhas sem balanço (ativo total ausente) são anos em que a empresa não está no painel."""
    d = df_trim[(df_trim['trimestre'] == 4) & df_trim['ativo_total'].notna()].copy()
    d = d.drop(columns=['trimestre', 'Data', 'passivo_mais_pl', 'ebitda_desp_fin', 'liquidez_corrente',
                        'divida_moeda_estrangeira'] + cfg.DESCARTAR_VAZIAS)
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- 2. empresas fora do Economatica (DFP da CVM)
COLS_FINANCEIRAS = ['ativo_total', 'ativo_circulante', 'passivo_circulante', 'patrimonio_liquido', 'caixa',
                    'emprestimos_cp', 'emprestimos_lp', 'divida_bruta', 'divida_liquida', 'reservas_lucro',
                    'receita', 'desp_financeira', 'ebit', 'ebitda', 'lucro_liquido_12m', 'caixa_operacional_12m']


def empresas_dfp(cnpjs_economatica, tickers_economatica, cad1):
    """Empresa-ano da DFP para as companhias NÃO financeiras que tiveram ações em bolsa e não estão no Economatica.

    São, na grande maioria, as que cancelaram o registro entre 2010 e 2026 (o viés de sobrevivência do Screening).
    Devolve (painel no formato de empresa_ano, mapa ticker -> CNPJ no formato de dados.mapa_ticker_cnpj).
    """
    from . import cvm_dfp
    p = cvm_dfp.carregar_painel()
    vm = dados.carregar_valor_mobiliario_cvm()
    tk = dados.ticker_por_cnpj(vm)

    financeiras = set(cad1.loc[cad1['financeira'], 'CNPJ_CIA'])
    universo = set(dados.acoes_em_bolsa(vm)['CNPJ_Companhia']) - set(cnpjs_economatica) - financeiras
    p = p[p['CNPJ_CIA'].isin(universo) & p['ano'].between(cfg.ANO_INI, cfg.ANO_FIM)].copy()

    # ticker: código do FCA (sem código nos FCAs antigos -> 'CVM' + código CVM); se já existe no Economatica
    # (código reaproveitado por outra empresa) ou repete, ganha sufixo
    mapa = pd.DataFrame({'CNPJ_CIA': sorted(p['CNPJ_CIA'].unique())})
    cod_cvm = cad1.set_index('CNPJ_CIA')['CD_CVM'].str.lstrip('0')
    mapa['ticker'] = mapa['CNPJ_CIA'].map(tk).fillna('CVM' + mapa['CNPJ_CIA'].map(cod_cvm).fillna(mapa['CNPJ_CIA']))
    usados = set(tickers_economatica)
    finais = []
    for t in mapa['ticker']:
        novo, k = t, 1
        while novo in usados:
            k += 1
            novo = f'{t}_{k}'
        usados.add(novo)
        finais.append(novo)
    mapa['ticker'] = finais
    mapa['como_casou'] = 'dfp'

    p = p.merge(mapa[['CNPJ_CIA', 'ticker']], on='CNPJ_CIA')
    return p[['ticker', 'ano'] + COLS_FINANCEIRAS], mapa[['ticker', 'CNPJ_CIA', 'como_casou']]


# ---------------------------------------------------------------- 3. CVM
def setor_macro(setor_ativ):
    """SETOR_ATIV da CVM -> macrossetor provisório (cfg.SETOR_MACRO).
    Sem setor na CVM (3 empresas, nenhum evento) -> 'Holdings e outros': uma categoria própria tão pequena não tem
    nenhum Y = 1 e causava separação perfeita na logística sem penalização (notebook 05)."""
    s = setor_ativ.str.replace(r'^Emp\. Adm\. Part\. - ', '', regex=True)
    return s.map(cfg.SETOR_MACRO).fillna('Holdings e outros')


def juntar_cvm(ea, mapa, cad1):
    """Traz CNPJ, situação, datas de registro/cancelamento e setor da CVM."""
    info = mapa.merge(cad1[['CNPJ_CIA', 'DENOM_SOCIAL', 'SIT', 'DT_REG_PRIMEIRO', 'DT_CANCEL',
                            'MOTIVO_CANCEL', 'SETOR_ATIV']], on='CNPJ_CIA', how='left')
    info['setor'] = setor_macro(info['SETOR_ATIV'])
    out = ea.merge(info, on='ticker', how='left')
    out['ano_registro'] = out['DT_REG_PRIMEIRO'].dt.year
    out['idade_registro'] = out['ano'] - out['ano_registro']
    out['pre_registro'] = out['idade_registro'] < 0
    return out


# ---------------------------------------------------------------- 4a. evento de crédito
def eventos_credito(ea, cad1):
    """Ano do primeiro evento de crédito de cada ticker (RJ, RE, falência, liquidação ou saída da bolsa em crise).

    Uma empresa conta como "em crise" se a situação de emissor, o motivo do cancelamento ou a razão social indicam
    crise (cfg.PADRAO_*). O ano do evento é o MENOR entre:
      - o ano de início da situação de emissor (RJ, falência...);
      - o ano do cancelamento do registro;
      - o primeiro ano sem balanço (último ano no painel + 1), se ela parou de publicar antes do fim dos dados.
    Mais a lista à mão (data/raw/eventos/eventos_credito.csv). Devolve um DataFrame por ticker.
    """
    ult = ea.groupby('ticker')['ano'].max()
    info = ea.drop_duplicates('ticker').set_index('ticker')[['CNPJ_CIA']].join(ult.rename('ultimo_ano'))
    c = cad1.set_index('CNPJ_CIA')[['DENOM_SOCIAL', 'SIT', 'DT_CANCEL', 'MOTIVO_CANCEL', 'SIT_EMISSOR',
                                    'DT_INI_SIT_EMISSOR']]
    info = info.join(c, on='CNPJ_CIA')

    sit = info['SIT_EMISSOR'].fillna('').str.upper().str.contains(cfg.PADRAO_SIT_CRISE)
    canc = (info['SIT'] == 'CANCELADA') & info['MOTIVO_CANCEL'].fillna('').str.upper().str.contains(cfg.PADRAO_CANCEL_CRISE)
    nome = info['DENOM_SOCIAL'].fillna('').str.upper().str.contains(cfg.PADRAO_NOME_CRISE)
    crise = sit | canc | nome

    ano_sit = pd.to_datetime(info['DT_INI_SIT_EMISSOR'], errors='coerce').dt.year.where(sit)
    ano_canc = info['DT_CANCEL'].dt.year.where(info['SIT'] == 'CANCELADA')
    parou = (info['ultimo_ano'] + 1).where(info['ultimo_ano'] < cfg.ULTIMO_ANO_DADOS)
    cand = pd.concat([ano_sit, ano_canc.where(crise), parou.where(crise)], axis=1)
    info['ano_evento_credito'] = cand.min(axis=1)
    s = info['SIT_EMISSOR'].fillna('')
    info['tipo_evento_credito'] = np.select(
        [sit & s.str.contains('FALID'), sit & s.str.contains('LIQUIDA'),
         sit & s.str.contains('EXTRAJUDICIAL'), sit & s.str.contains('RECUPERA'),
         canc & info['MOTIVO_CANCEL'].str.contains('LIQUIDA', na=False), canc, nome],
        ['falência', 'liquidação', 'RE', 'RJ', 'liquidação', 'cancelamento de ofício', 'crise (razão social)'],
        default=None)
    info['fonte_evento'] = np.where(crise, 'CVM', None)

    # lista conferida à mão: casa pelo ticker da base ou, se o ticker for outro (ex.: GOLL4), pelo CNPJ do FCA;
    # vale a data mais antiga
    man = pd.read_csv(cfg.ARQ_EVENTOS, dtype=str)
    man['ano'] = pd.to_datetime(man['data_evento'], errors='coerce').dt.year
    vm = dados.carregar_valor_mobiliario_cvm().dropna(subset=['ticker'])
    cnpj_fca = vm.drop_duplicates('ticker', keep='last').set_index('ticker')['CNPJ_Companhia']
    ticker_da_base = info.reset_index().dropna(subset=['CNPJ_CIA']).drop_duplicates('CNPJ_CIA').set_index('CNPJ_CIA')['ticker']
    man['ticker'] = np.where(man['ticker'].isin(info.index), man['ticker'],
                             man['ticker'].map(cnpj_fca).map(ticker_da_base))
    man = man.dropna(subset=['ano', 'ticker']).sort_values('ano').drop_duplicates('ticker').set_index('ticker')
    novo = info.loc[man.index, 'ano_evento_credito'].isna() | (man['ano'] < info.loc[man.index, 'ano_evento_credito'])
    idx = man.index[novo.to_numpy()]
    info.loc[idx, 'ano_evento_credito'] = man.loc[idx, 'ano']
    info.loc[idx, 'tipo_evento_credito'] = man.loc[idx, 'tipo']
    info.loc[idx, 'fonte_evento'] = 'lista manual'

    out = info[info['ano_evento_credito'].notna()]
    return out[['ano_evento_credito', 'tipo_evento_credito', 'fonte_evento']].reset_index()


def adicionar_evento_credito(ea, cad1):
    ev = eventos_credito(ea, cad1)
    out = ea.drop(columns=[c for c in ev.columns if c != 'ticker' and c in ea.columns]).merge(ev, on='ticker', how='left')
    out['ano_evento_credito'] = out['ano_evento_credito'].astype(float)
    return out


# ---------------------------------------------------------------- 4b. alvo
def regra_evento(ano, ano_evento, h):
    """(evento em t+1..t+h, evento em t ou antes). Sem evento ou com cfg.EVENTO_CREDITO_NO_ALVO = False: tudo False."""
    if not cfg.EVENTO_CREDITO_NO_ALVO or ano_evento is None:
        z = np.zeros(len(ano), dtype=bool)
        return z, z
    ev = np.asarray(ano_evento, dtype=float)
    ano = np.asarray(ano, dtype=float)
    return (ev > ano) & (ev <= ano + h), ev <= ano


def adicionar_alvo(ea):
    """cobertura, distress, Y e a marcação da amostra em risco.

    - cobertura_t = EBIT_t / despesa financeira_t; se a despesa for <= 0 (contaminação cambial) -> NaN.
    - distress_t = 1 se cobertura_t < 1 (EBIT negativo também conta).
    - Y_t = 1 se distress em t+1 E t+2; 0 se t+1 e t+2 são classificáveis e não ambos em distress; NaN caso contrário.
    - Y_t = 1 também se há evento de crédito em t+1 ou t+2 (cfg.EVENTO_CREDITO_NO_ALVO), mesmo sem balanço;
      quem já teve o evento em t ou antes sai da amostra em risco.
    O t+1 e o t+2 são procurados por ANO (não pela próxima linha), para não pular buracos no painel.
    """
    ea = ea.copy()
    desp = ea['desp_financeira'].where(ea['desp_financeira'] > 0)
    ea['cobertura'] = ea['ebit'] / desp
    ea['distress'] = np.where(ea['cobertura'].isna(), np.nan, (ea['cobertura'] < 1).astype(float))

    chave = ea.set_index(['ticker', 'ano'])
    for k in (1, 2):
        idx = pd.MultiIndex.from_arrays([ea['ticker'], ea['ano'] + k])
        ea[f'distress_t{k}'] = chave['distress'].reindex(idx).to_numpy()
        ea[f'presente_t{k}'] = idx.isin(chave.index)

    d1, d2 = ea['distress_t1'], ea['distress_t2']
    y_cob = np.where(d1.notna() & d2.notna(), ((d1 == 1) & (d2 == 1)).astype(float), np.nan)
    na_janela, ja_teve = regra_evento(ea['ano'], ea.get('ano_evento_credito'), 2)
    ea['Y'] = np.where(na_janela, 1.0, y_cob)
    ea['origem_Y'] = np.select([na_janela & (y_cob == 1), na_janela, y_cob == 1], ['cobertura e evento', 'evento', 'cobertura'],
                               default=None)

    ea['em_risco'] = ((ea['cobertura'] >= 1) & ea['Y'].notna() & ~ja_teve
                      & ea['ano'].between(cfg.ANO_INI, cfg.ANO_T_MAX))
    return ea


def funil_amostra(ea):
    """Quantas linhas caem em cada filtro até a amostra em risco (t de 2010 a 2023)."""
    d = ea[ea['ano'].between(cfg.ANO_INI, cfg.ANO_T_MAX)]
    passos = [('empresa-anos com balanço, t = 2010–2023', pd.Series(True, index=d.index))]
    m = passos[0][1]
    m = m & d['cobertura'].notna();             passos.append(('cobertura classificável em t', m))
    m = m & (d['cobertura'] >= 1);              passos.append(('fora de distress em t (cobertura ≥ 1)', m))
    m = m & d['Y'].notna();                     passos.append(('Y observado (t+1 e t+2 classificáveis ou evento de crédito)', m))
    m = m & d['em_risco'];                      passos.append(('sem evento de crédito em t ou antes', m))
    out = pd.DataFrame({'etapa': [p[0] for p in passos], 'linhas': [int(p[1].sum()) for p in passos]})
    out['perdidas na etapa'] = (-out['linhas'].diff()).fillna(0).astype(int)
    return out


# ---------------------------------------------------------------- 5. explicativas
def adicionar_explicativas(ea):
    ea = ea.copy()
    ea['alavancagem'] = _div(ea['divida_bruta'], ea['ativo_total'])
    ea['divida_cp_share'] = _div(ea['emprestimos_cp'], ea['emprestimos_cp'] + ea['emprestimos_lp'])
    ea['roa'] = _div(ea['lucro_liquido_12m'], ea['ativo_total'])
    ea['liquidez'] = _div(ea['caixa'], ea['passivo_circulante'])
    ea['sem_divida'] = ea['divida_bruta'] == 0
    ea['fco_divida'] = _div(ea['caixa_operacional_12m'], ea['divida_bruta'])
    ea['log_ativo'] = np.log(ea['ativo_total'].where(ea['ativo_total'] > 0))
    return ea


# ---------------------------------------------------------------- 6. Z''-score
def adicionar_zscore(ea):
    """Z'' de Altman para mercados emergentes. Maior risco = Z'' menor.
    X2 usa reservas de lucro COM o sinal: no Economatica, negativo = prejuízos acumulados maiores que as reservas."""
    ea = ea.copy()
    at = ea['ativo_total']
    ea['z_x1'] = _div(ea['ativo_circulante'] - ea['passivo_circulante'], at)
    ea['z_x2'] = _div(ea['reservas_lucro'], at)
    ea['z_x3'] = _div(ea['ebit'], at)
    passivo = (at - ea['patrimonio_liquido']).where(lambda s: s > 0)
    ea['z_x4'] = _div(ea['patrimonio_liquido'], passivo)
    ea['z_score'] = 3.25 + 6.56 * ea['z_x1'] + 3.26 * ea['z_x2'] + 6.72 * ea['z_x3'] + 1.05 * ea['z_x4']
    return ea


# ---------------------------------------------------------------- 7. divisão
def adicionar_split(ea):
    ea = ea.copy()
    ea['split'] = pd.NA
    for nome, (a, b) in cfg.SPLITS.items():
        ea.loc[ea['ano'].between(a, b), 'split'] = nome
    return ea


# ---------------------------------------------------------------- 8. winsorização
def limites_winsor(ea, cols):
    """Percentis 1% e 99% calculados SÓ no treino (amostra em risco)."""
    treino = ea[(ea['split'] == 'treino') & ea['em_risco']]
    return pd.DataFrame({c: treino[c].quantile(list(cfg.WINSOR)).to_numpy() for c in cols},
                        index=['p_inf', 'p_sup']).T


def aplicar_winsor(ea, limites):
    """Cria colunas <var>_w cortadas nos limites do treino (aplicados a todas as linhas)."""
    ea = ea.copy()
    for c, (lo, hi) in limites.iterrows():
        ea[f'{c}_w'] = ea[c].clip(lo, hi)
    return ea


# ---------------------------------------------------------------- 9. dicionário
DICIONARIO = {
    'ticker': 'Código de negociação na B3 (sem o sufixo <XBSP> do Economatica; nas empresas da DFP, o código do FCA)',
    'fonte': 'Economatica (empresas listadas hoje) ou CVM (DFP) (empresas fora do Screening, quase todas canceladas)',
    'ano_evento_credito': 'Ano do 1º evento de crédito (RJ, RE, falência, liquidação ou saída da bolsa em crise); NaN se nenhum',
    'tipo_evento_credito': 'Tipo do evento de crédito',
    'fonte_evento': 'De onde veio o evento: CVM (cadastro) ou lista manual (data/raw/eventos)',
    'origem_Y': 'Por que Y = 1: cobertura (2 anos < 1), evento (de crédito em t+1..t+2) ou os dois',
    'ano': 'Ano t (dados do 4º trimestre = ano completo)',
    'CNPJ_CIA': 'CNPJ da companhia (ligado pelo FCA da CVM; ver como_casou)',
    'como_casou': 'Como o ticker foi ligado ao CNPJ: exato, raiz (4 letras), manual, manual_conferir, sem_match',
    'DENOM_SOCIAL': 'Razão social no cadastro da CVM',
    'SIT': 'Situação atual do registro na CVM',
    'DT_REG_PRIMEIRO': 'Primeira data de registro na CVM',
    'DT_CANCEL': 'Data de cancelamento do registro (se houver)',
    'MOTIVO_CANCEL': 'Motivo do cancelamento (CVM)',
    'SETOR_ATIV': 'Setor de atividade da CVM (original)',
    'setor': 'Macrossetor PROVISÓRIO (agrupamento do SETOR_ATIV; será trocado pelo Subsetor Bovespa)',
    'ano_registro': 'Ano do primeiro registro na CVM',
    'idade_registro': 't − ano de registro (negativo = demonstração anterior ao registro, ex.: prospecto de IPO)',
    'pre_registro': 'True se t é anterior ao ano de registro na CVM',
    'ativo_total': 'Ativo total (R$ mil, na data)',
    'ativo_circulante': 'Ativo circulante (R$ mil)',
    'passivo_circulante': 'Passivo circulante (R$ mil)',
    'patrimonio_liquido': 'Patrimônio líquido (R$ mil)',
    'caixa': 'Caixa e equivalentes (R$ mil)',
    'emprestimos_cp': 'Empréstimos e financiamentos de curto prazo (R$ mil)',
    'emprestimos_lp': 'Empréstimos e financiamentos de longo prazo (R$ mil)',
    'divida_bruta': 'Dívida total bruta = empréstimos CP + LP (R$ mil)',
    'divida_liquida': 'Dívida total líquida (Economatica, R$ mil)',
    'reservas_lucro': 'Reservas de lucro (R$ mil). Negativo = prejuízos acumulados maiores que as reservas (plano padronizado do Economatica)',
    'receita': 'Receita líquida do ano (R$ mil)',
    'desp_financeira': 'Despesa financeira do ano (R$ mil, positiva = despesa)',
    'ebit': 'EBIT 12 meses (R$ mil). SÓ usado no alvo e no Z\'\'',
    'ebitda': 'EBITDA 12 meses (R$ mil). Não usado',
    'lucro_liquido_12m': 'Lucro líquido 12 meses (R$ mil)',
    'caixa_operacional_12m': 'Fluxo de caixa operacional 12 meses (R$ mil; exportado em R$ e convertido)',
    'valor_mercado': 'Valor de mercado da empresa (R$ mil). Não usado nas explicativas',
    'cobertura': 'EBIT / despesa financeira em t; NaN se despesa ≤ 0 ou ausente',
    'distress': '1 se cobertura < 1; 0 se ≥ 1; NaN se não classificável',
    'distress_t1': 'distress em t+1 (procurado por ano)',
    'distress_t2': 'distress em t+2',
    'presente_t1': 'Empresa tem balanço em t+1',
    'presente_t2': 'Empresa tem balanço em t+2',
    'Y': 'ALVO: 1 se distress em t+1 e t+2 OU evento de crédito em t+1..t+2; 0 se ambos classificáveis e nenhum dos dois; NaN caso contrário',
    'em_risco': 'True se está na amostra de modelagem: cobertura_t ≥ 1, Y observado, sem evento de crédito até t e t ≤ 2023',
    'alavancagem': 'H1 (+): dívida bruta / ativo total',
    'divida_cp_share': 'H1 (+): empréstimos CP / (CP + LP); NaN se sem dívida',
    'roa': 'H2 (−): lucro líquido 12m / ativo total',
    'liquidez': 'H3 (−): caixa / passivo circulante',
    'fco_divida': 'H4 (−): caixa operacional 12m / dívida bruta; NaN se dívida = 0',
    'sem_divida': 'True se dívida bruta = 0 (fco_divida e divida_cp_share ficam NaN)',
    'log_ativo': 'Controle de tamanho: log(ativo total em R$ mil)',
    'z_x1': "Z'': (AC − PC) / AT",
    'z_x2': "Z'': reservas de lucro / AT",
    'z_x3': "Z'': EBIT / AT",
    'z_x4': "Z'': PL / (AT − PL); NaN se passivo ≤ 0",
    'z_score': "Z''-score (Altman, emergentes) = 3,25 + 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4; menor = mais risco",
    'split': 'treino (t 2010–2018), validacao (2019–2020), teste (2021–2023)',
}


def escrever_dicionario(ea, caminho):
    linhas = ['# Dicionário de variáveis — base_empresa_ano.parquet', '',
              'Uma linha por empresa × ano (4º trimestre). Valores em R$ mil, salvo indicação.', '',
              '| coluna | tipo | % NaN | descrição |', '|---|---|---|---|']
    for c in ea.columns:
        desc = DICIONARIO.get(c)
        if desc is None and c.endswith('_w'):
            desc = f'{c[:-2]} winsorizada nos percentis 1% e 99% do treino'
        pct = f'{100 * ea[c].isna().mean():.1f}'.replace('.', ',')
        linhas.append(f'| `{c}` | {ea[c].dtype} | {pct}% | {desc or ""} |')
    caminho.write_text('\n'.join(linhas) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- pipeline
def construir():
    """Roda a Fase 2 inteira a partir de data/raw/. Devolve (base, info) com os relatórios intermediários."""
    trim = dados.carregar_economatica()
    ea = empresa_ano(trim)
    info = {'empresa_anos_4t': len(ea), 'empresas_4t': ea['ticker'].nunique()}

    fin = ea['ticker'].isin(cfg.EXCLUIR_FINANCEIRAS)
    info['excluidas_financeiras'] = (int(fin.sum()), sorted(ea.loc[fin, 'ticker'].unique()))
    ea = ea[~fin]

    cad1 = dados.cad_por_cnpj(dados.carregar_cad_cvm())
    mapa = dados.mapa_ticker_cnpj(sorted(ea['ticker'].unique()))
    info['mapa'] = mapa
    ea['fonte'] = 'Economatica'

    if cfg.INCLUIR_DFP:
        extra, mapa_dfp = empresas_dfp(mapa['CNPJ_CIA'].dropna(), ea['ticker'].unique(), cad1)
        extra['fonte'] = 'CVM (DFP)'
        info['dfp'] = {'empresas': extra['ticker'].nunique(), 'empresa_anos': len(extra)}
        ea = pd.concat([ea, extra], ignore_index=True)
        mapa = pd.concat([mapa, mapa_dfp], ignore_index=True)
    ea = juntar_cvm(ea, mapa, cad1)

    ea = adicionar_evento_credito(ea, cad1)
    ea = adicionar_alvo(ea)
    ea = adicionar_explicativas(ea)
    ea = adicionar_zscore(ea)
    ea = adicionar_split(ea)
    lim = limites_winsor(ea, list(cfg.EXPLICATIVAS))
    info['limites_winsor'] = lim
    ea = aplicar_winsor(ea, lim)
    return ea.sort_values(['ticker', 'ano']).reset_index(drop=True), info
