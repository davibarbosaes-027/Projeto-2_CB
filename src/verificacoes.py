"""Verificações da Fase 1 (inspeção). Só leem a base; não transformam nada."""
import numpy as np
import pandas as pd


def resumo_arquivo(df_bruto, nome):
    """Formato, nº de colunas, empresas e períodos de uma exportação bruta."""
    return {
        'arquivo': nome,
        'linhas': len(df_bruto),
        'colunas': df_bruto.shape[1],
        'empresas (Ativo)': df_bruto['Ativo'].nunique(),
        'períodos (Data)': df_bruto['Data'].nunique(),
        'primeiro período': df_bruto['Data'].iloc[0],
        'último período': df_bruto['Data'].iloc[-1],
        'duplicatas Ativo+Data': int(df_bruto.duplicated(['Ativo', 'Data']).sum()),
    }


def padrao_acumulado(df, variaveis):
    """Testa se a variável se comporta como acumulado no ano ('no exercício').

    Para cada empresa-ano com os 4 trimestres positivos:
      - razão mediana T2/T1, T3/T1, T4/T1 (acumulado no ano -> ~2, ~3, ~4; 12 meses -> ~1);
      - % de empresa-anos com T1 < T2 < T3 < T4;
      - % de viradas de ano em que o T1 seguinte cai em relação ao T4 anterior.
    """
    linhas = []
    for v in variaveis:
        p = df.pivot_table(index=['ticker', 'ano'], columns='trimestre', values=v)
        completos = p.dropna()
        completos = completos[(completos > 0).all(axis=1)]
        cresce = ((completos[2] > completos[1]) & (completos[3] > completos[2]) & (completos[4] > completos[3]))

        t4 = p[4].rename('t4').reset_index()
        t1_seg = p[1].rename('t1_seg').reset_index().assign(ano=lambda t: t['ano'] - 1)
        virada = t4.merge(t1_seg, on=['ticker', 'ano']).dropna()
        virada = virada[(virada['t4'] > 0) & (virada['t1_seg'] > 0)]

        linhas.append({
            'variável': v,
            'empresa-anos': len(completos),
            'T2/T1': completos[2].div(completos[1]).median(),
            'T3/T1': completos[3].div(completos[1]).median(),
            'T4/T1': completos[4].div(completos[1]).median(),
            '% T1<T2<T3<T4': cresce.mean(),
            '% T1 seguinte < T4': (virada['t1_seg'] < virada['t4']).mean(),
        })
    out = pd.DataFrame(linhas).set_index('variável')
    out['parece acumulada'] = (out['T4/T1'] > 2.5) & (out['% T1<T2<T3<T4'] > 0.8)
    return out


def sinal(df, var, filtro=None):
    """Contagem de valores positivos, zero, negativos e ausentes."""
    s = df[var] if filtro is None else df.loc[filtro, var]
    return pd.Series({'> 0': (s > 0).sum(), '= 0': (s == 0).sum(), '< 0': (s < 0).sum(), 'NaN': s.isna().sum()})


def mudancas_entre_trimestres(df, var):
    """% de vezes que a variável muda em relação ao trimestre anterior, por trimestre.
    Uma reserva de balanço (como reservas de lucro) só deve mudar de verdade no 4º trimestre."""
    d = df[['ticker', 'ano', 'trimestre', var]].dropna().sort_values(['ticker', 'ano', 'trimestre'])
    d['anterior'] = d.groupby('ticker')[var].shift()
    d = d.dropna(subset=['anterior'])
    return d.assign(mudou=d[var] != d['anterior']).groupby('trimestre')['mudou'].mean()


def primeiro_ultimo_balanco(df, var='ativo_total'):
    """Primeiro e último ano com balanço (var não nula) de cada empresa."""
    d = df[df[var].notna()]
    return d.groupby('ticker').agg(primeiro_ano=('ano', 'min'), ultimo_ano=('ano', 'max'),
                                   n_trimestres=('ano', 'size'))
