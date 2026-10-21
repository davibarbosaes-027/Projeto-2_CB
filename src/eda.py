"""Fase 3: cálculos e gráficos da análise exploratória (todos com a identidade visual de src/style.py)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as cfg, dados, style

PADRAO_DISTRESS = r'RECUPERA|FAL[EÊI]|LIQUIDA'   # sinais de distress no nome ou no motivo


# ---------------------------------------------------------------- viés de sobrevivência
def cnpjs_com_acoes_em_bolsa():
    """CNPJs que tiveram ações (ON, PN ou units) negociadas em bolsa em algum FCA de 2010 a 2026."""
    vm = dados.carregar_valor_mobiliario_cvm()
    acoes = vm['Valor_Mobiliario'].str.contains('Ações|Units', na=False) & (vm['Mercado'] == 'Bolsa')
    return set(vm.loc[acoes, 'CNPJ_Companhia'])


def ativas_cvm_por_ano(cad, anos):
    """CNPJs ativos na CVM em cada ano: DT_REG ≤ 31/12 e (sem cancelamento ou cancelamento depois de 31/12).
    Usa todas as linhas do cadastro (empresas que se registraram mais de uma vez contam se algum registro vale)."""
    out = {}
    for a in anos:
        fim = pd.Timestamp(f'{a}-12-31')
        ok = (cad['DT_REG'] <= fim) & (cad['DT_CANCEL'].isna() | (cad['DT_CANCEL'] > fim))
        out[a] = set(cad.loc[ok, 'CNPJ_CIA'])
    return out


def cobertura_por_ano(ea, cad, em_bolsa):
    """Nº de empresas na base × nº de companhias ativas na CVM (não financeiras), e a cobertura."""
    cad_nf = cad[~cad['financeira']]
    anos = range(cfg.ANO_INI, 2026)
    ativas = ativas_cvm_por_ano(cad_nf, anos)
    linhas = []
    for a in anos:
        base_a = ea[ea['ano'] == a]
        cnpjs_base = set(base_a['CNPJ_CIA'].dropna())
        cvm_a = ativas[a]
        cvm_bolsa = cvm_a & em_bolsa
        linhas.append({
            'ano': a,
            'base': base_a['ticker'].nunique(),
            'base (já registradas)': int((~base_a['pre_registro'].astype(bool)).sum()),
            'CVM ativas': len(cvm_a),
            'CVM ativas c/ ações em bolsa': len(cvm_bolsa),
            'cobertura CVM': len(cnpjs_base & cvm_a) / len(cvm_a),
            'cobertura bolsa': len(cnpjs_base & cvm_bolsa) / len(cvm_bolsa),
        })
    return pd.DataFrame(linhas).set_index('ano')


def canceladas_fora_da_base(ea, cad, em_bolsa):
    """Companhias não financeiras canceladas entre 2010 e 2026 que não estão na base."""
    c = cad[(cad['SIT'] == 'CANCELADA') & cad['DT_CANCEL'].dt.year.between(cfg.ANO_INI, 2026) & ~cad['financeira']]
    c = c.sort_values('DT_CANCEL').drop_duplicates('CNPJ_CIA', keep='last')
    c = c[~c['CNPJ_CIA'].isin(set(ea['CNPJ_CIA'].dropna()))].copy()
    c['ano_cancel'] = c['DT_CANCEL'].dt.year
    c['tinha_acoes_em_bolsa'] = c['CNPJ_CIA'].isin(em_bolsa)
    texto = c['DENOM_SOCIAL'].fillna('') + ' ' + c['MOTIVO_CANCEL'].fillna('') + ' ' + c['SIT_EMISSOR'].fillna('')
    c['sinal_distress'] = texto.str.upper().str.contains(PADRAO_DISTRESS) | c['MOTIVO_CANCEL'].str.contains('OFÍCIO', na=False)
    return c[['CNPJ_CIA', 'DENOM_SOCIAL', 'DT_REG', 'DT_CANCEL', 'ano_cancel', 'MOTIVO_CANCEL', 'SETOR_ATIV',
              'tinha_acoes_em_bolsa', 'sinal_distress']]


# ---------------------------------------------------------------- análise por quintil
def taxa_por_quintil(df, var, y='Y', q=5):
    """Taxa de evento por quintil de var. Os cortes são calculados em df (o treino)."""
    d = df[[var, y]].dropna()
    d['quintil'] = pd.qcut(d[var], q, labels=[f'Q{i}' for i in range(1, q + 1)], duplicates='drop')
    g = d.groupby('quintil', observed=True)
    return g[y].agg(taxa='mean', n='size', eventos='sum').join(g[var].agg(de='min', ate='max'))


def veredito(tab, sinal):
    """Compara o último e o primeiro quintil com o sinal esperado: confirma, contraria ou sem padrão."""
    dif = tab['taxa'].iloc[-1] - tab['taxa'].iloc[0]
    if abs(dif) < 0.03:
        return 'sem padrão claro', dif
    esperado = 1 if sinal == '+' else -1
    return ('confirma' if np.sign(dif) == esperado else 'contraria'), dif


def vif(df, cols):
    X = df[cols].dropna()
    X = (X - X.mean()) / X.std()
    X.insert(0, 'const', 1.0)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=cols, name='VIF')


# ---------------------------------------------------------------- gráficos
def fig_cobertura(cob):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.TAMANHO_FIG, gridspec_kw={'width_ratios': [1.35, 1]})
    anos = cob.index
    ax1.plot(anos, cob['CVM ativas'], color=cfg.CINZA, lw=2.5)
    ax1.plot(anos, cob['CVM ativas c/ ações em bolsa'], color='#8C8C8C', lw=2.5, ls='--')
    ax1.plot(anos, cob['base'], color=cfg.COR_DESTAQUE, lw=3)
    x = anos[-1] + 0.3
    ax1.text(x, cob['CVM ativas'].iloc[-1], 'CVM: ativas', color='#8C8C8C', va='center', fontsize=10)
    ax1.text(x, cob['CVM ativas c/ ações em bolsa'].iloc[-1], 'CVM: com ações\nem bolsa', color='#8C8C8C', va='center', fontsize=10)
    ax1.text(x, cob['base'].iloc[-1], 'Base', color=cfg.COR_DESTAQUE, va='center', fontsize=10, fontweight='bold')
    ax1.set_xlim(anos[0] - 0.5, anos[-1] + 3.2)
    for ax in (ax1, ax2):
        ax.set_xticks(range(anos[0], anos[-1] + 1, 3))
    ax1.set_ylim(0, None)
    ax1.yaxis.set_major_formatter(style.formatador(0))
    ax1.set_title('Nº de companhias não financeiras por ano', fontsize=11, color=cfg.COR_EIXO, loc='left')

    ax2.plot(anos, cob['cobertura bolsa'], color=cfg.COR_DESTAQUE, lw=3, marker='o', ms=4)
    for a in [anos[0], 2015, 2020, anos[-1]]:
        ax2.annotate(style.pct_br(cob.loc[a, 'cobertura bolsa'], 0), (a, cob.loc[a, 'cobertura bolsa']),
                     xytext=(0, 8), textcoords='offset points', ha='center', fontsize=10, color=cfg.COR_EIXO)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(style.formatador(0, pct=True))
    ax2.set_title('Cobertura: % das listadas que estão na base', fontsize=11, color=cfg.COR_EIXO, loc='left')
    return fig


def fig_barras_ano(serie, destaque=None, pct=False, casas=0):
    """Barras por ano com rótulo direto; anos em `destaque` em laranja, resto em cinza."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    destaque = destaque or []
    cores = [cfg.COR_DESTAQUE if a in destaque else cfg.CINZA for a in serie.index]
    b = ax.bar(serie.index, serie.values, color=cores, width=0.7)
    style.rotular_barras(ax, b, casas=casas, pct=pct)
    ax.set_xticks(serie.index)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    return fig, ax


def fig_quintil(tab, extra=None):
    """Taxa de evento por quintil (barras); `extra` = barra adicional (ex.: 'sem dívida')."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    rotulos = list(tab.index.astype(str))
    valores = list(tab['taxa'])
    if extra is not None:
        rotulos.append(extra[0])
        valores.append(extra[1])
    maxv = np.nanmax(valores)
    cores = [cfg.COR_DESTAQUE if v == maxv else cfg.CINZA for v in valores]
    if extra is not None:
        cores[-1] = cfg.CINZA_2
    b = ax.bar(rotulos, valores, color=cores, width=0.65)
    style.rotular_barras(ax, b, casas=1, pct=True)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    return fig, ax


def fig_heatmap(mat, cmap, vmin=None, vmax=None, casas=2, pct=False, destaque=None, fontsize=9):
    """Mapa de calor com valores anotados em pt-BR. `destaque`: máscara booleana de células a realçar."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0 if mat.shape[1] <= 8 else 45,
                  ha='center' if mat.shape[1] <= 8 else 'right')
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            txt = style.pct_br(v, casas) if pct else style.num_br(v, casas)
            negrito = destaque is not None and destaque.values[i, j]
            ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize,
                    color=cfg.COR_TITULO, fontweight='bold' if negrito else 'normal')
    return fig, ax
