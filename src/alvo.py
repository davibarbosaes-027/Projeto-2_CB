"""Etapa B: sensibilidade da definição do alvo (numerador da cobertura e nº de anos em distress)."""
import numpy as np
import pandas as pd

from . import config as cfg


def cobertura(ea, numerador):
    """numerador / despesa financeira; despesa <= 0 ou ausente -> NaN (não classificável)."""
    desp = ea['desp_financeira'].where(ea['desp_financeira'] > 0)
    return ea[numerador] / desp


def calcular(ea, nome):
    """Alvo pela definição `nome` de cfg.DEFINICOES_ALVO. Devolve um DataFrame alinhado a `ea` com
    cob, distress, Y e em_risco. A amostra em risco exige: fora de distress em t pela MESMA definição,
    Y observado, t <= último ano de dados − horizonte e ano posterior ao registro na CVM (sem pré-IPO)."""
    num, h = cfg.DEFINICOES_ALVO[nome]
    out = pd.DataFrame(index=ea.index)
    out['cob'] = cobertura(ea, num)
    out['distress'] = np.where(out['cob'].isna(), np.nan, (out['cob'] < 1).astype(float))

    chave = pd.Series(out['distress'].to_numpy(), index=pd.MultiIndex.from_arrays([ea['ticker'], ea['ano']]))
    futuros = []
    for k in range(1, h + 1):
        idx = pd.MultiIndex.from_arrays([ea['ticker'], ea['ano'] + k])
        futuros.append(chave.reindex(idx).to_numpy())
    f = np.column_stack(futuros)
    observado = ~np.isnan(f).any(axis=1)
    out['Y'] = np.where(observado, (f == 1).all(axis=1).astype(float), np.nan)

    out['em_risco'] = ((out['cob'] >= 1) & out['Y'].notna() & ~ea['pre_registro'].astype(bool)
                       & ea['ano'].between(cfg.ANO_INI, cfg.ULTIMO_ANO_DADOS - h))
    return out


def prevalencia_por_ano(ea, numerador):
    """% das empresas classificáveis com cobertura < 1, por ano (sem os anos pré-registro)."""
    d = ea[~ea['pre_registro'].astype(bool)]
    c = cobertura(d, numerador)
    return (c[c.notna()] < 1).groupby(d.loc[c.notna(), 'ano']).mean()


def prevalencia_persistente_por_ano(ea, numerador, anos=3):
    """% das empresas com cobertura < 1 em t e nos `anos`−1 anos anteriores (todos classificáveis).
    É a medida comparável à de empresas "zumbi" do BIS (cobertura < 1 por 3 anos seguidos)."""
    d = ea[~ea['pre_registro'].astype(bool)]
    c = cobertura(d, numerador)
    dist = pd.Series(np.where(c.isna(), np.nan, (c < 1).astype(float)),
                     index=pd.MultiIndex.from_arrays([d['ticker'], d['ano']]))
    hist = np.column_stack([dist.reindex(pd.MultiIndex.from_arrays([d['ticker'], d['ano'] - k])).to_numpy()
                            for k in range(anos)])
    ok = ~np.isnan(hist).any(axis=1)
    pers = pd.Series((hist[ok] == 1).all(axis=1), index=d.index[ok])
    return pers.groupby(d.loc[pers.index, 'ano']).mean()


def resumo(ea, nome, alvo):
    """Tamanho da amostra em risco, eventos e taxa, total e por split, e eventos por parâmetro no treino."""
    r = ea[alvo['em_risco']].assign(Y=alvo.loc[alvo['em_risco'], 'Y'])
    num, h = cfg.DEFINICOES_ALVO[nome]
    linha = {'definição': nome, 'cobertura': f'{num.upper()} / desp. fin.', 'anos em distress': h,
             't até': cfg.ULTIMO_ANO_DADOS - h,
             'prevalência média (cob < 1)': prevalencia_por_ano(ea, num).mean(),
             'amostra em risco': len(r), 'eventos': int(r['Y'].sum()), 'taxa de eventos': r['Y'].mean()}
    for sp in ['treino', 'validacao', 'teste']:
        g = r[r['split'] == sp]
        linha[f'n {sp}'] = len(g)
        linha[f'eventos {sp}'] = int(g['Y'].sum())
        linha[f'taxa {sp}'] = g['Y'].mean() if len(g) else np.nan
    linha['eventos por parâmetro (treino)'] = linha['eventos treino'] / cfg.N_PARAMETROS_PREVISTOS
    return linha


def taxa_eventos_por_ano(ea, alvo):
    r = alvo[alvo['em_risco']]
    return r['Y'].groupby(ea.loc[r.index, 'ano']).mean()


# ---------------------------------------------------------------- eventos reais de crédito
def carregar_eventos():
    ev = pd.read_csv(cfg.ARQ_EVENTOS, dtype=str)
    ev['data_evento'] = pd.to_datetime(ev['data_evento'], errors='coerce')
    ev['ano_evento'] = ev['data_evento'].dt.year.astype('Int64')
    return ev


def eventos_cvm(ea, cad1):
    """Candidatos a evento tirados do cadastro da CVM: empresas da base cuja situação de emissor é
    recuperação judicial/extrajudicial ou falência, com a data de início da situação.
    Atenção: DT_INI_SIT_EMISSOR é o início da situação ATUAL, que pode não ser a data do pedido."""
    c = cad1[cad1['CNPJ_CIA'].isin(set(ea['CNPJ_CIA'].dropna()))
             & cad1['SIT_EMISSOR'].str.contains('RECUPERA|FAL', na=False)].copy()
    tk = ea.dropna(subset=['CNPJ_CIA']).drop_duplicates('CNPJ_CIA').set_index('CNPJ_CIA')['ticker']
    c['ticker'] = c['CNPJ_CIA'].map(tk)
    c['tipo'] = np.where(c['SIT_EMISSOR'].str.contains('EXTRAJUDICIAL'), 'RE',
                         np.where(c['SIT_EMISSOR'].str.contains('FAL'), 'falência', 'RJ'))
    c['data_evento'] = pd.to_datetime(c['DT_INI_SIT_EMISSOR'], errors='coerce')
    c['ano_evento'] = c['data_evento'].dt.year.astype('Int64')
    c['empresa'] = c['DENOM_SOCIAL']
    return c[['ticker', 'empresa', 'data_evento', 'ano_evento', 'tipo']].sort_values('data_evento').reset_index(drop=True)


def status_eventos(ea, eventos, numerador):
    """Para cada evento: a empresa estava em distress (cobertura < 1) no ano do evento, em t−1 e em t−2?
    'sim' / 'não' / 'n.c.' (despesa financeira <= 0 ou ausente) / 'sem dado' (empresa fora da base no ano)."""
    cob = cobertura(ea, numerador)
    tab = pd.Series(cob.to_numpy(), index=pd.MultiIndex.from_arrays([ea['ticker'], ea['ano']]))
    presentes = set(tab.index)

    def st(t, a):
        if pd.isna(a) or (t, a) not in presentes:
            return 'sem dado'
        v = tab[(t, a)]
        return 'n.c.' if pd.isna(v) else ('sim' if v < 1 else 'não')

    linhas = []
    for _, e in eventos.iterrows():
        a = e['ano_evento']
        linhas.append({'ticker': e['ticker'], 'tipo': e['tipo'], 'ano_evento': a,
                       'no ano': st(e['ticker'], a),
                       't−1': st(e['ticker'], a - 1 if pd.notna(a) else a),
                       't−2': st(e['ticker'], a - 2 if pd.notna(a) else a)})
    out = pd.DataFrame(linhas)
    out['sinaliza em t−1 ou t−2'] = (out['t−1'] == 'sim') | (out['t−2'] == 'sim')
    out['avaliável'] = out[['t−1', 't−2']].isin(['sim', 'não']).any(axis=1)
    return out
