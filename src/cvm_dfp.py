"""Painel empresa-ano a partir da DFP da CVM (Dados Abertos), para TODAS as companhias abertas,
inclusive as que já cancelaram o registro.

Por que existe: o Screening do Economatica só trouxe empresas listadas hoje (viés de sobrevivência). As que
quebraram e saíram da bolsa entre 2010 e 2026 ficaram de fora, e são justamente os casos que o modelo
precisa ver. A DFP traz as demonstrações anuais delas até o último ano em que publicaram.

As contas seguem o plano padronizado da CVM para empresas NÃO financeiras (o mesmo em 2010–2025):
    1        Ativo Total                       2.01     Passivo Circulante
    1.01     Ativo Circulante                  2.01.04  Empréstimos e Financiamentos (CP)
    1.01.01  Caixa e Equivalentes              2.02.01  Empréstimos e Financiamentos (LP)
    1.01.02  Aplicações Financeiras            2.03     Patrimônio Líquido (consolidado: inclui minoritários)
    2.03.04  Reservas de Lucros                2.03.05  Lucros/Prejuízos Acumulados
    3.01     Receita                           3.05     Resultado antes do resultado financeiro e dos tributos (EBIT)
    3.06.02  Despesas Financeiras (negativa)   3.11     Lucro/Prejuízo do período
    6.01     Caixa Líquido Atividades Operacionais (DFC)
    7.04.01  Depreciação, Amortização e Exaustão (DVA)
Como no Economatica: consolidado, com fallback para o individual quando a empresa não publica consolidado.
"""
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import config as cfg

URL_DFP = 'https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip'

CONTAS = {
    'BPA': {'1': 'ativo_total', '1.01': 'ativo_circulante', '1.01.01': 'caixa', '1.01.02': 'aplicacoes_cp'},
    'BPP': {'2.01': 'passivo_circulante', '2.01.04': 'emprestimos_cp', '2.02.01': 'emprestimos_lp',
            '2.03': 'pl_total', '2.03.09': 'pl_minoritarios', '2.03.04': 'reservas_lucro'},
    'DRE': {'3.01': 'receita', '3.05': 'ebit', '3.06.02': 'desp_fin_bruta', '3.11': 'lucro_total',
            '3.11.01': 'lucro_controladores'},
    'DFC_MI': {'6.01': 'caixa_operacional_12m'},
    'DFC_MD': {'6.01': 'caixa_operacional_12m'},
    'DVA': {'7.04.01': 'depreciacao_dva'},
}
# Depreciação na DFC (método indireto): linhas 6.01.01.xx cujo nome fala em depreciação/amortização/exaustão.
# Bate melhor com o EBITDA do Economatica (90% das empresas-ano a menos de 1%) do que a DVA; a DVA fica de reserva.
PADRAO_DEPREC = r'(?i)deprecia|amortiza|exaust'
# Descrição esperada do EBIT: confirma que é o plano das não financeiras (seguradoras usam outro plano).
DS_EBIT = 'Resultado Antes do Resultado Financeiro e dos Tributos'


# ---------------------------------------------------------------- download
def baixar_dfp(anos=None, forcar=False):
    """Baixa os zips da DFP que ainda não estão em data/raw/cvm/dfp/ (dado público, não versionado no git)."""
    anos = anos or range(cfg.ANO_INI, cfg.ANO_FIM + 1)
    cfg.DIR_DFP_CVM.mkdir(parents=True, exist_ok=True)
    for a in anos:
        destino = cfg.DIR_DFP_CVM / f'dfp_{a}.zip'
        if forcar or not destino.exists() or destino.stat().st_size == 0:
            urllib.request.urlretrieve(URL_DFP.format(ano=a), destino)


# ---------------------------------------------------------------- leitura
def _ler(arq, ano, demo, tipo):
    """Uma demonstração (ex.: BPA consolidado) do exercício corrente, só nas contas de CONTAS."""
    nome = f'dfp_cia_aberta_{demo}_{tipo}_{ano}.csv'
    if nome not in arq.namelist():
        return pd.DataFrame()
    t = pd.read_csv(arq.open(nome), sep=';', encoding='latin-1', dtype=str,
                    usecols=['CNPJ_CIA', 'DENOM_CIA', 'VERSAO', 'ORDEM_EXERC', 'DT_FIM_EXERC', 'ESCALA_MOEDA',
                             'CD_CONTA', 'DS_CONTA', 'VL_CONTA'])
    dep = (t['CD_CONTA'].str.startswith('6.01.01.') & t['DS_CONTA'].str.contains(PADRAO_DEPREC, na=False)
           if demo == 'DFC_MI' else False)
    t = t[t['ORDEM_EXERC'].str.upper().str.startswith('ÚLT') & (t['CD_CONTA'].isin(CONTAS[demo]) | dep)].copy()
    if t.empty:
        return t
    t['VERSAO'] = t['VERSAO'].astype(int)
    t = t[t['VERSAO'] == t.groupby('CNPJ_CIA')['VERSAO'].transform('max')]   # reapresentações: última versão
    v = pd.to_numeric(t['VL_CONTA'], errors='coerce')
    t['valor'] = np.where(t['ESCALA_MOEDA'].str.upper() == 'UNIDADE', v / 1000, v)   # tudo em R$ mil
    t['ano'] = pd.to_datetime(t['DT_FIM_EXERC'], errors='coerce').dt.year
    t['conta'] = t['CD_CONTA'].map(CONTAS[demo]).fillna('depreciacao_dfc')
    t['valor'] = np.where(t['conta'] == 'depreciacao_dfc', t['valor'].abs(), t['valor'])
    t['tipo'] = tipo
    t['ebit_ok'] = (t['CD_CONTA'] == '3.05') & (t['DS_CONTA'].str.strip() == DS_EBIT)
    return t[['CNPJ_CIA', 'DENOM_CIA', 'ano', 'tipo', 'conta', 'valor', 'ebit_ok']]


def ler_ano(ano):
    """Empresa-ano da DFP de um ano: uma linha por CNPJ, consolidado se houver, senão individual."""
    with zipfile.ZipFile(cfg.DIR_DFP_CVM / f'dfp_{ano}.zip') as arq:
        longo = pd.concat([_ler(arq, ano, d, tp) for d in CONTAS for tp in ('con', 'ind')], ignore_index=True)
    if longo.empty:
        return pd.DataFrame()

    # consolidado tem prioridade: a empresa usa o individual só se não tem ativo total consolidado
    tem_con = set(longo.loc[(longo['tipo'] == 'con') & (longo['conta'] == 'ativo_total'), 'CNPJ_CIA'])
    longo = longo[(longo['tipo'] == 'con') == longo['CNPJ_CIA'].isin(tem_con)]

    nao_fin = set(longo.loc[longo['ebit_ok'], 'CNPJ_CIA'])   # plano das não financeiras
    longo = longo[longo['CNPJ_CIA'].isin(nao_fin)]
    # DFC_MI e DFC_MD têm a mesma conta 6.01 (método indireto ou direto): uma delas basta ('first');
    # a depreciação pode estar em várias linhas da DFC e é somada
    dep = longo['conta'] == 'depreciacao_dfc'
    largo = longo[~dep].pivot_table(index=['CNPJ_CIA', 'ano'], columns='conta', values='valor', aggfunc='first')
    largo = largo.join(longo[dep].groupby(['CNPJ_CIA', 'ano'])['valor'].sum().rename('depreciacao_dfc'))
    info = longo.groupby(['CNPJ_CIA', 'ano']).agg(denom_dfp=('DENOM_CIA', 'first'), consolidado=('tipo', 'first'))
    out = info.join(largo).reset_index()
    out['consolidado'] = out['consolidado'] == 'con'
    return out


def montar_painel(anos=None):
    """Todas as DFPs de 2010 em diante, com as colunas no mesmo formato da base do Economatica (R$ mil)."""
    anos = anos or range(cfg.ANO_INI, cfg.ANO_FIM + 1)
    baixar_dfp(anos)
    p = pd.concat([ler_ano(a) for a in anos], ignore_index=True)
    # o arquivo de um ano pode trazer exercícios encerrados em outra data; fica a última leitura de cada CNPJ-ano
    p = p.drop_duplicates(['CNPJ_CIA', 'ano'], keep='last')

    for c in set().union(*[set(v.values()) for v in CONTAS.values()]) | {'depreciacao_dfc'}:
        if c not in p:
            p[c] = np.nan
    p['ano'] = p['ano'].astype(int)
    p['divida_bruta'] = p[['emprestimos_cp', 'emprestimos_lp']].sum(axis=1, min_count=1)
    p['divida_liquida'] = p['divida_bruta'] - p[['caixa', 'aplicacoes_cp']].sum(axis=1, min_count=1)
    # Como no Economatica: PL e lucro dos CONTROLADORES (sem minoritários). No individual não há essa divisão.
    p['patrimonio_liquido'] = p['pl_total'] - p['pl_minoritarios'].fillna(0)
    p['lucro_liquido_12m'] = p['lucro_controladores'].fillna(p['lucro_total'])
    p['desp_financeira'] = -p['desp_fin_bruta']                 # a DRE traz a despesa negativa
    p['depreciacao'] = p['depreciacao_dfc'].fillna(p['depreciacao_dva'].abs())
    p['ebitda'] = p['ebit'] + p['depreciacao']
    return p.sort_values(['CNPJ_CIA', 'ano']).reset_index(drop=True)


def comparar_com_economatica(ea, painel, cols=None):
    """Para as empresas-ano que estão nas DUAS fontes (casadas por CNPJ e ano): % com diferença menor que 1% e 5%,
    e a concordância na classificação de distress (cobertura < 1) com EBIT e com EBITDA."""
    cols = cols or ['ativo_total', 'ativo_circulante', 'passivo_circulante', 'patrimonio_liquido', 'caixa',
                    'divida_bruta', 'reservas_lucro', 'receita', 'ebit', 'ebitda', 'desp_financeira',
                    'lucro_liquido_12m', 'caixa_operacional_12m']
    m = ea.dropna(subset=['CNPJ_CIA']).merge(painel, on=['CNPJ_CIA', 'ano'], suffixes=('_e', '_c'))
    linhas = []
    for c in cols:
        a, b = m[f'{c}_e'], m[f'{c}_c']
        ok = a.notna() & b.notna()
        dif = (b[ok] - a[ok]).abs() / a[ok].abs().clip(lower=1)
        linhas.append({'variável': c, 'pares': int(ok.sum()), 'dif. < 1%': (dif < 0.01).mean(),
                       'dif. < 5%': (dif < 0.05).mean()})
    for num in ['ebit', 'ebitda']:
        dist = {}
        for s in ('_e', '_c'):
            desp = m[f'desp_financeira{s}'].where(m[f'desp_financeira{s}'] > 0)
            cob = m[f'{num}{s}'] / desp
            dist[s] = np.where(cob.isna(), np.nan, (cob < 1).astype(float))
        ok = ~np.isnan(dist['_e']) & ~np.isnan(dist['_c'])
        linhas.append({'variável': f'distress ({num.upper()} / desp. fin. < 1)', 'pares': int(ok.sum()),
                       'dif. < 1%': (dist['_e'][ok] == dist['_c'][ok]).mean(), 'dif. < 5%': np.nan})
    out = pd.DataFrame(linhas).set_index('variável')
    out.attrs['empresas'] = m['CNPJ_CIA'].nunique()
    return out


def carregar_painel(refazer=False):
    """Painel da DFP com cache em data/processed/ (ler os zips leva alguns minutos)."""
    if cfg.ARQ_DFP_PROCESSADO.exists() and not refazer:
        return pd.read_parquet(cfg.ARQ_DFP_PROCESSADO)
    p = montar_painel()
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    p.to_parquet(cfg.ARQ_DFP_PROCESSADO, index=False)
    return p
