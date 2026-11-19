"""Fase 4: métricas, limiar, bootstrap por empresa e gráficos da avaliação dos modelos.

Métrica principal: AUC-PR (average precision). Com 11,5% de eventos, prever sempre "não" acerta 88%, então a
acurácia não entra. A AUC-PR de um modelo sem informação é igual à taxa de eventos (a "linha de base" dos gráficos).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score)

from . import config as cfg, style


# ---------------------------------------------------------------- métricas
def recall_topk(y, p, k=cfg.TOP_K):
    """Fração dos eventos capturada ao marcar os k% de maior risco (empates resolvidos pela ordem)."""
    y, p = np.asarray(y), np.asarray(p)
    n_marc = max(1, int(np.ceil(k * len(y))))
    marcados = np.argsort(-p, kind='stable')[:n_marc]
    return y[marcados].sum() / y.sum() if y.sum() else np.nan


def escolher_limiar(y, p, beta=cfg.BETA_LIMIAR):
    """Limiar que maximiza o F-beta (beta > 1 privilegia o recall: deixar passar um distress custa mais que um alarme falso).
    Deve ser chamado com previsões FORA DA AMOSTRA (dobras da validação temporal), nunca com o teste."""
    prec, rec, lim = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / np.where(beta ** 2 * prec + rec == 0, np.nan, beta ** 2 * prec + rec)
    return float(lim[np.nanargmax(f)])


def metricas(y, p, limiar=None, beta=cfg.BETA_LIMIAR):
    """AUC-PR, AUC-ROC, Brier, recall no top-k e, se houver limiar, precisão/recall/F-beta naquele corte."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {'n': len(y), 'eventos': int(y.sum()), 'taxa': y.mean(),
           'AUC-PR': average_precision_score(y, p) if y.sum() else np.nan,
           'AUC-ROC': roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           'Brier': brier_score_loss(y, np.clip(p, 0, 1)),
           f'recall top {int(100 * cfg.TOP_K)}%': recall_topk(y, p)}
    if limiar is not None:
        pred = p >= limiar
        tp = int((pred & (y == 1)).sum())
        prec = tp / pred.sum() if pred.sum() else np.nan
        rec = tp / y.sum() if y.sum() else np.nan
        f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec) if tp else 0.0
        out.update({'limiar': limiar, 'marcados': int(pred.sum()), 'precisão': prec, 'recall': rec, f'F{beta:g}': f})
    return out


def tabela_metricas(y, previsoes, limiares=None):
    """Uma linha por modelo. `previsoes` = {nome: probabilidades}; `limiares` = {nome: limiar} (opcional)."""
    limiares = limiares or {}
    tab = pd.DataFrame({nome: metricas(y, p, limiares.get(nome)) for nome, p in previsoes.items()}).T
    for c in ('n', 'eventos', 'marcados'):          # contagens inteiras (senão saem como 286,000 com o formato de 3 casas)
        if c in tab:
            tab[c] = pd.array(tab[c].astype(float).round().to_numpy(), dtype='Int64') if tab[c].notna().all() else tab[c]
    return tab


# ---------------------------------------------------------------- bootstrap por empresa
def bootstrap_empresas(y, previsoes, grupos, referencia, n_boot=cfg.N_BOOT, seed=cfg.SEED):
    """Reamostra EMPRESAS (cluster) com reposição e recalcula AUC-PR e AUC-ROC de cada modelo na mesma amostra.
    Reamostrar linhas soltas subestimaria a incerteza, porque a mesma empresa aparece em vários anos.
    Devolve (resumo, diferenças): o resumo traz mediana e intervalo; as diferenças comparam cada modelo com `referencia`
    (ex.: o Z''), com a fração das reamostragens em que o modelo supera a referência."""
    y = np.asarray(y).astype(int)
    grupos = np.asarray(grupos)
    P = {k: np.asarray(v, dtype=float) for k, v in previsoes.items()}
    idx_por_grupo = [np.flatnonzero(grupos == g) for g in pd.unique(grupos)]
    rng = np.random.default_rng(seed)
    amostras = {k: {'AUC-PR': [], 'AUC-ROC': []} for k in P}
    for _ in range(n_boot):
        sel = np.concatenate([idx_por_grupo[i] for i in rng.integers(0, len(idx_por_grupo), len(idx_por_grupo))])
        yb = y[sel]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        for k, p in P.items():
            amostras[k]['AUC-PR'].append(average_precision_score(yb, p[sel]))
            amostras[k]['AUC-ROC'].append(roc_auc_score(yb, p[sel]))
    lo, hi = cfg.IC
    linhas, difs = [], []
    for k, a in amostras.items():
        for m, v in a.items():
            v = np.array(v)
            linhas.append({'modelo': k, 'métrica': m, 'mediana': np.median(v),
                           'ic_inf': np.quantile(v, lo), 'ic_sup': np.quantile(v, hi)})
            if k != referencia:
                d = v - np.array(amostras[referencia][m])
                difs.append({'modelo': k, 'métrica': m, 'diferença mediana': np.median(d),
                             'ic_inf': np.quantile(d, lo), 'ic_sup': np.quantile(d, hi), f'P(> {referencia})': (d > 0).mean()})
    return pd.DataFrame(linhas), pd.DataFrame(difs)


def se_auc_pr(y, p, grupos, n_boot=200, seed=cfg.SEED):
    """Erro-padrão (bootstrap por empresa) da AUC-PR. Usado na regra de 1 erro-padrão da escolha de C."""
    res, _ = bootstrap_empresas(y, {'m': p}, grupos, 'm', n_boot=n_boot, seed=seed)
    v = res[res['métrica'] == 'AUC-PR']
    return float((v['ic_sup'].iloc[0] - v['ic_inf'].iloc[0]) / (2 * 1.645))   # IC de 90% ≈ ±1,645 EP


def calibracao(y, p, bins=5):
    """Taxa observada × probabilidade média prevista por quantil de previsão."""
    d = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    d['bin'] = pd.qcut(d['p'].rank(method='first'), bins, labels=False)
    return d.groupby('bin').agg(prevista=('p', 'mean'), observada=('y', 'mean'), n=('y', 'size'))


# ---------------------------------------------------------------- gráficos
def _cor(i, destaque):
    return cfg.COR_DESTAQUE if i == destaque else ['#8C8C8C', cfg.CINZA, '#5A5A5A', cfg.CINZA_2, '#A8A8A8'][i % 5]


def fig_pr(y, previsoes, destaque):
    """Curvas precisão × recall; a linha tracejada é a AUC-PR de um modelo sem informação (a taxa de eventos)."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    y = np.asarray(y)
    for i, (nome, p) in enumerate(previsoes.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=3.2 if nome == destaque else 2, color=cfg.COR_DESTAQUE if nome == destaque else _cor(i + 1, -1),
                label=f'{nome} ({style.num_br(average_precision_score(y, p), 2)})')
    ax.axhline(y.mean(), color=cfg.COR_EIXO, lw=1, ls='--')
    ax.text(0.005, y.mean() - 0.012, f'sem informação ({style.pct_br(y.mean(), 1)})', ha='left', va='top', fontsize=9, color=cfg.COR_EIXO)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Recall (eventos capturados)')
    ax.set_ylabel('Precisão')
    ax.xaxis.set_major_formatter(style.formatador(0, pct=True))
    ax.yaxis.set_major_formatter(style.formatador(0, pct=True))
    ax.legend(loc='upper right', fontsize=9.5, title='modelo (AUC-PR)', title_fontsize=9.5)
    return fig, ax


def fig_ic(resumo, metrica='AUC-PR', destaque=None, taxa=None):
    """Mediana e intervalo (bootstrap por empresa) da métrica de cada modelo."""
    v = resumo[resumo['métrica'] == metrica].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    y = np.arange(len(v))[::-1]
    for yi, (_, r) in zip(y, v.iterrows()):
        cor = cfg.COR_DESTAQUE if r['modelo'] == destaque else '#8C8C8C'
        ax.plot([r['ic_inf'], r['ic_sup']], [yi, yi], color=cor, lw=3, solid_capstyle='round')
        ax.plot(r['mediana'], yi, 'o', color=cor, ms=9)
        ax.annotate(style.num_br(r['mediana'], 2), (r['mediana'], yi), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=9.5, color=cfg.COR_EIXO)
    if taxa is not None:
        ax.axvline(taxa, color=cfg.COR_EIXO, lw=1, ls='--')
        ax.text(taxa, len(v) - 1.5, ' sem informação', fontsize=9, color=cfg.COR_EIXO, va='center', ha='left')
    ax.set_yticks(y, v['modelo'])
    ax.set_xlabel(metrica)
    ax.xaxis.set_major_formatter(style.formatador(2))
    ax.grid(axis='y', visible=False)
    return fig, ax


def fig_coeficientes(tab):
    """Razão de chances por +1 desvio-padrão (escala log), com IC 90% agrupado por empresa. Laranja = confirma a hipótese."""
    t = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    for yi, (nome, r) in enumerate(t.iterrows()):
        cor = cfg.COR_DESTAQUE if r.get('veredito') == 'confirma' else '#8C8C8C'
        ax.plot([r['or_inf'], r['or_sup']], [yi, yi], color=cor, lw=3, solid_capstyle='round')
        ax.plot(r['odds_ratio'], yi, 'o', color=cor, ms=9)
        ax.annotate(style.num_br(r['odds_ratio'], 2), (r['odds_ratio'], yi), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=9.5, color=cfg.COR_EIXO)
    ax.axvline(1, color=cfg.COR_EIXO, lw=1)
    ax.set_xscale('log')
    ax.set_yticks(range(len(t)), t['rótulo'])
    ax.set_xlabel('Razão de chances por +1 desvio-padrão (escala log)')
    ax.xaxis.set_major_formatter(style.formatador(1))
    ax.xaxis.set_minor_formatter(style.formatador(1))
    ax.grid(axis='y', visible=False)
    return fig, ax


def fig_calibracao(y, previsoes, destaque):
    """Probabilidade prevista × taxa observada, em quintis de previsão."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    teto = 0.05
    for i, (nome, p) in enumerate(previsoes.items()):
        c = calibracao(y, p)
        teto = max(teto, c['prevista'].max(), c['observada'].max())
        ax.plot(c['prevista'], c['observada'], 'o-', lw=3 if nome == destaque else 1.8, ms=7,
                color=cfg.COR_DESTAQUE if nome == destaque else _cor(i + 1, -1), label=nome)
    ax.plot([0, teto * 1.05], [0, teto * 1.05], color=cfg.COR_EIXO, lw=1, ls='--')
    ax.set_xlim(0, teto * 1.05)
    ax.set_ylim(0, teto * 1.05)
    ax.set_xlabel('Probabilidade média prevista')
    ax.set_ylabel('Taxa observada de eventos')
    ax.xaxis.set_major_formatter(style.formatador(0, pct=True))
    ax.yaxis.set_major_formatter(style.formatador(0, pct=True))
    ax.legend(loc='upper left', fontsize=9.5)
    return fig, ax


ROTULOS_IMP = {'fco_divida': 'FCO / dívida', 'divida_cp_share': 'Dívida de curto prazo', 'roa': 'ROA',
               'liquidez': 'Liquidez', 'alavancagem': 'Alavancagem', 'setor': 'Setor', 'log_ativo': 'Tamanho',
               'sem_divida': 'Sem dívida (flag)'}


def fig_importancia(imp, destaque=None):
    """Um painel por modelo (colunas de `imp`), mesma escala; ordem das variáveis pela do primeiro modelo."""
    imp = imp.sort_values(imp.columns[0])
    fig, axs = plt.subplots(1, imp.shape[1], figsize=cfg.TAMANHO_FIG, sharey=True)
    lo, hi = min(0, imp.min().min()) * 1.15, imp.max().max() * 1.15
    for ax, col in zip(np.atleast_1d(axs), imp.columns):
        ax.barh(range(len(imp)), imp[col], color=cfg.COR_DESTAQUE if col == destaque else '#8C8C8C', height=0.8)
        ax.axvline(0, color=cfg.COR_EIXO, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_title(col, loc='left', fontsize=12)
        ax.set_xlabel('Queda da AUC-ROC ao embaralhar')
        ax.xaxis.set_major_formatter(style.formatador(2))
        ax.grid(axis='y', visible=False)
    np.atleast_1d(axs)[0].set_yticks(range(len(imp)), [ROTULOS_IMP.get(v, v) for v in imp.index])
    return fig, axs


def fig_residuos(paineis):
    """`paineis` = {título: saída de modelagem.residuos_por_faixa}. Resíduo médio por faixa com banda de ±2 EP."""
    fig, axs = plt.subplots(1, len(paineis), figsize=cfg.TAMANHO_FIG, sharey=True)
    for i, (ax, (tit, r)) in enumerate(zip(np.atleast_1d(axs), paineis.items())):
        ax.fill_between(r['p'], -2 * r['ep'], 2 * r['ep'], color=cfg.CINZA_GRID, label='±2 erros-padrão')
        ax.plot(r['p'], r['residuo'], 'o-', color=cfg.COR_DESTAQUE, lw=3, ms=8)
        ax.axhline(0, color=cfg.COR_EIXO, lw=1)
        ax.set_title(tit, loc='left', fontsize=12)
        ax.set_xlabel('Probabilidade ajustada (média da faixa)')
        ax.xaxis.set_major_formatter(style.formatador(0, pct=True))
        ax.yaxis.set_major_formatter(style.formatador(1, pct=True))
        if i == 0:
            ax.set_ylabel('Resíduo médio (observado − ajustado)')
            ax.legend(loc='upper left', fontsize=9.5)
    return fig, axs


def fig_selecao_c(tab, c_escolhido):
    """AUC-PR fora da amostra (validação temporal no treino) em função de C."""
    fig, ax = plt.subplots(figsize=cfg.TAMANHO_FIG)
    ax.plot(tab['C'], tab['AUC-PR'], 'o-', color='#8C8C8C', lw=2.5, ms=7)
    esc = tab[tab['C'] == c_escolhido].iloc[0]
    ax.plot(esc['C'], esc['AUC-PR'], 'o', color=cfg.COR_DESTAQUE, ms=13)
    ax.annotate(f"C = {style.num_br(c_escolhido, 2)}", (esc['C'], esc['AUC-PR']), xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=11, color=cfg.COR_DESTAQUE, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlabel('C (inverso da regularização L2; menor = mais regularizado)')
    ax.set_ylabel('AUC-PR fora da amostra')
    ax.yaxis.set_major_formatter(style.formatador(2))
    return fig, ax