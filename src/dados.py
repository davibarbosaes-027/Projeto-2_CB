"""Leitura dos dados brutos (Economatica e CVM). Nada aqui altera os arquivos em data/raw/."""
import zipfile

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------- Economatica
def ler_economatica_bruto(caminho):
    """Lê um CSV do Economatica como veio. '-' e vazio viram NaN (nunca zero)."""
    return pd.read_csv(caminho, encoding='latin-1', na_values=['-', ''], keep_default_na=True)


def _nome_curto(col):
    return cfg.RENOMEAR.get(col.split('|')[0].strip(), col)


def carregar_economatica(trimestral=True):
    """Junta as 3 exportações por Ativo + Data numa base empresa × trimestre.

    - outer join: nenhuma linha é perdida (as exportações têm conjuntos de tickers ligeiramente diferentes);
    - nomes de coluna curtos (cfg.RENOMEAR);
    - colunas exportadas em R$ são passadas para milhares, como o resto;
    - ticker sem o sufixo da bolsa ('PETR4<XBSP>' -> 'PETR4'), ano e trimestre numéricos.
    """
    bases = [ler_economatica_bruto(f).rename(columns=_nome_curto) for f in cfg.ARQ_ECONOMATICA]
    df = bases[0]
    for b in bases[1:]:
        df = df.merge(b, on=['Ativo', 'Data'], how='outer')

    for c in cfg.EM_REAIS:
        df[c] = df[c] / 1000

    df['ticker'] = df['Ativo'].str.replace(r'<.*>', '', regex=True)
    df['ano'] = df['Data'].str[2:].astype(int)
    df['trimestre'] = df['Data'].str[0].astype(int)
    df = df.drop(columns='Ativo')

    chaves = ['ticker', 'Data', 'ano', 'trimestre']
    df = df[chaves + [c for c in df.columns if c not in chaves]]
    return df.sort_values(['ticker', 'ano', 'trimestre']).reset_index(drop=True)


# ---------------------------------------------------------------- CVM
def carregar_cad_cvm():
    """Cadastro de companhias abertas (Dados Abertos CVM)."""
    cad = pd.read_csv(cfg.ARQ_CAD_CVM, sep=';', encoding='latin-1', dtype=str)
    for c in ['DT_REG', 'DT_CANCEL', 'DT_INI_SIT', 'DT_CONST']:
        cad[c] = pd.to_datetime(cad[c], errors='coerce')
    cad['financeira'] = cad['SETOR_ATIV'].isin(cfg.SETORES_FINANCEIROS)
    return cad


def carregar_valor_mobiliario_cvm():
    """Valores mobiliários do FCA (todos os anos): é onde a CVM liga o código de negociação ao CNPJ."""
    partes = []
    for z in sorted(cfg.DIR_FCA_CVM.glob('fca_*.zip')):
        with zipfile.ZipFile(z) as arq:
            nome = next(n for n in arq.namelist() if 'valor_mobiliario' in n)
            t = pd.read_csv(arq.open(nome), sep=';', encoding='latin-1', dtype=str)
        t['ano_fca'] = int(z.stem[-4:])
        partes.append(t)
    vm = pd.concat(partes, ignore_index=True)
    vm['ticker'] = vm['Codigo_Negociacao'].str.strip().str.upper()
    return vm


def mapa_ticker_cnpj(tickers):
    """Liga cada ticker do Economatica a um CNPJ da CVM, em 3 etapas, guardando como casou:

    1. 'exato'  : código de negociação idêntico no FCA;
    2. 'raiz'   : mesmas 4 primeiras letras (ex.: MRSA3B -> MRSA3), só se a raiz levar a um único CNPJ;
    3. 'manual' : tabela cfg.MAPA_MANUAL_CNPJ (conferida pelo nome no cadastro).
    O que sobra fica sem CNPJ ('sem_match').
    """
    vm = carregar_valor_mobiliario_cvm().dropna(subset=['ticker'])
    vm = vm.sort_values('ano_fca')
    exato = vm.drop_duplicates('ticker', keep='last').set_index('ticker')['CNPJ_Companhia']

    vm['raiz'] = vm['ticker'].str[:4]
    por_raiz = vm.groupby('raiz')['CNPJ_Companhia'].agg(lambda s: s.unique().tolist())

    linhas = []
    for t in tickers:
        if t in cfg.MAPA_MANUAL_CNPJ:
            cnpj, conf, _ = cfg.MAPA_MANUAL_CNPJ[t]
            linhas.append((t, cnpj, 'manual' if conf == 'ok' else 'manual_conferir'))
        elif t in exato.index:
            linhas.append((t, exato[t], 'exato'))
        elif len(por_raiz.get(t[:4], [])) == 1:
            linhas.append((t, por_raiz[t[:4]][0], 'raiz'))
        else:
            linhas.append((t, np.nan, 'sem_match'))
    return pd.DataFrame(linhas, columns=['ticker', 'CNPJ_CIA', 'como_casou'])


def cad_por_cnpj(cad):
    """Uma linha por CNPJ. O cadastro repete CNPJ quando a empresa se registrou mais de uma vez;
    ficamos com o registro mais recente (é o que vale para a situação atual) e guardamos o
    primeiro DT_REG (para a idade de registro)."""
    cad = cad.sort_values(['CNPJ_CIA', 'DT_REG'])
    primeiro_reg = cad.groupby('CNPJ_CIA')['DT_REG'].min().rename('DT_REG_PRIMEIRO')
    ultimo = cad.drop_duplicates('CNPJ_CIA', keep='last').set_index('CNPJ_CIA')
    return ultimo.join(primeiro_reg).reset_index()
