"""Fase 4: amostra de modelagem, pipelines sem vazamento, validação temporal e inferência da logística.

Regra de ouro: tudo que "aprende" com os dados (limites da winsorização, p99 do sem-dívida, medianas, média e desvio
da padronização, coeficientes) é ajustado SÓ no treino da dobra. Por isso essas etapas são transformadores de um
`Pipeline`, e não colunas pré-calculadas na base (as `*_w` da Fase 2 servem só à exploração).
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import alvo as alvo_mod, avaliacao, config as cfg


# ---------------------------------------------------------------- amostra
def carregar_base():
    arq = cfg.DATA_PROCESSED / 'base_empresa_ano.parquet'
    if not arq.exists():
        raise FileNotFoundError(
            f'{arq} não existe. Essa base não é versionada (.gitignore). Duas saídas: (1) copie o arquivo da pessoa da dupla '
            'que já rodou o notebooks/02_base.ipynb para data/processed/; ou (2) coloque os CSVs do Economatica em '
            'data/raw/economatica/ (nomes em cfg.ARQ_ECONOMATICA) e rode o notebooks/02_base.ipynb.')
    return pd.read_parquet(arq)


def juntar_macro(df):
    """Junta as variáveis macro anuais (cfg.MACRO_COLS) pelo ano t. Use só valores conhecidos no fim de t."""
    if not cfg.MACRO_COLS:
        return df
    if not cfg.ARQ_MACRO.exists():
        raise FileNotFoundError(f'cfg.MACRO_COLS = {cfg.MACRO_COLS}, mas {cfg.ARQ_MACRO} não existe. '
                                'Crie o CSV com a coluna `ano` e uma coluna por variável.')
    m = pd.read_csv(cfg.ARQ_MACRO)[['ano'] + list(cfg.MACRO_COLS)]
    out = df.merge(m, on='ano', how='left')
    faltam = out.loc[out['em_risco'], list(cfg.MACRO_COLS)].isna().any(axis=1).sum()
    if faltam:
        raise ValueError(f'{faltam} linhas da amostra em risco sem valor macro: o CSV precisa cobrir todos os anos t.')
    return out


def preparar_amostra(ea, definicao=None):
    """Amostra em risco para a definição de alvo `definicao` (cfg.DEFINICOES_ALVO), com Y inteiro.

    Usa src/alvo.py (mesma regra do notebook 04: fora de distress em t pela MESMA definição, Y observado, sem
    pré-registro, t até ULTIMO_ANO_DADOS − horizonte). Mantém só as linhas com divisão temporal definida."""
    definicao = definicao or cfg.ALVO_MODELO
    a = alvo_mod.calcular(ea, definicao)
    d = ea.drop(columns=[c for c in ('Y', 'em_risco', 'distress') if c in ea.columns]).copy()
    d['Y'] = a['Y']
    d['em_risco'] = a['em_risco']
    d = juntar_macro(d)
    d = d[d['em_risco'] & d['split'].notna()].copy()
    d['Y'] = d['Y'].astype(int)
    d['split'] = d['split'].astype(str)
    d.attrs['definicao'] = definicao
    return d.reset_index(drop=True)


def colunas_modelo(usar_setor=True):
    return (list(cfg.FEATURES_NUM) + list(cfg.FEATURES_FLAG) + list(cfg.MACRO_COLS)
            + ([cfg.FEATURE_SETOR] if usar_setor else []))


def colunas_entrada():
    """Tudo o que algum modelo lê (as features + z_score, usado pela baseline)."""
    return list(dict.fromkeys(colunas_modelo() + ['z_score']))


def resumo_amostra(amostra):
    """Tamanho e eventos por divisão, e eventos por parâmetro previsto (critério 3 do notebook 04)."""
    g = amostra.groupby('split')['Y'].agg(n='size', eventos='sum', taxa='mean').reindex(['treino', 'validacao', 'teste'])
    tot = pd.DataFrame({'n': [len(amostra)], 'eventos': [amostra['Y'].sum()], 'taxa': [amostra['Y'].mean()]}, index=['total'])
    out = pd.concat([g, tot])
    out['empresas'] = [amostra.loc[amostra['split'] == s, 'ticker'].nunique() for s in g.index] + [amostra['ticker'].nunique()]
    out['eventos por parâmetro'] = np.where(out.index == 'treino', out['eventos'] / cfg.N_PARAMETROS_PREVISTOS, np.nan)
    return out


# ---------------------------------------------------------------- transformadores (ajustados só no treino)
class Winsorizar(BaseEstimator, TransformerMixin):
    """Corta as colunas `cols` nos percentis `q` do conjunto de ajuste (mesma regra de base.limites_winsor)."""

    def __init__(self, cols=None, q=cfg.WINSOR):
        self.cols = cols
        self.q = q

    def fit(self, X, y=None):
        cols = list(cfg.FEATURES_NUM if self.cols is None else self.cols)
        self.cols_ = [c for c in cols if c in X.columns]
        self.limites_ = {c: tuple(X[c].quantile(list(self.q))) for c in self.cols_}
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.limites_.items():
            X[c] = X[c].clip(lo, hi)
        return X


class AgruparSetores(BaseEstimator, TransformerMixin):
    """Junta em 'Outros setores' os setores com menos de `min_eventos` eventos NO CONJUNTO DE AJUSTE (ou setor ausente).
    Um setor sem eventos (ex.: 'Sem classificação') faz a logística sem penalização divergir (separação perfeita), e com
    poucos eventos o coeficiente não é estimável com sentido. Se o próprio 'Outros setores' ainda ficar abaixo do mínimo,
    essas empresas são somadas ao maior setor. Categorias nunca vistas no ajuste (teste) também vão para 'Outros'."""

    def __init__(self, coluna=cfg.FEATURE_SETOR, min_eventos=cfg.MIN_EVENTOS_SETOR, rotulo='Outros setores'):
        self.coluna = coluna
        self.min_eventos = min_eventos
        self.rotulo = rotulo

    def fit(self, X, y=None):
        y = pd.Series(np.asarray(y), index=X.index)
        g = pd.DataFrame({'n': X[self.coluna].value_counts(), 'ev': y.groupby(X[self.coluna]).sum()}).fillna(0)
        self.mantidos_ = set(g.index[g['ev'] >= self.min_eventos])
        raros = g.loc[~g.index.isin(self.mantidos_)]
        self.agrupados_ = sorted(raros.index.astype(str))
        self.destino_ = self.rotulo
        if len(raros) and raros['ev'].sum() < self.min_eventos and self.mantidos_:
            self.destino_ = g.loc[list(self.mantidos_), 'n'].idxmax()      # ainda poucos eventos: soma ao maior setor
        self.tabela_ = g
        return self

    def transform(self, X):
        X = X.copy()
        X[self.coluna] = X[self.coluna].where(X[self.coluna].isin(self.mantidos_), self.destino_)
        return X


class RegraSemDivida(BaseEstimator, TransformerMixin):
    """Empresas sem dívida têm NaN estrutural em fco_divida e divida_cp_share (a flag `sem_divida` guarda a informação).

    regra='p99_zero': fco_divida recebe o teto do treino (o p99 da winsorização: capacidade de pagamento "infinita")
                      e divida_cp_share recebe 0. É a sugestão do notebook 03 (seção 3).
    regra='nenhuma' : deixa NaN; a imputação pela mediana (próxima etapa) cuida do resto."""

    def __init__(self, regra='p99_zero'):
        self.regra = regra

    def fit(self, X, y=None):
        self.teto_fco_ = X['fco_divida'].max() if 'fco_divida' in X else np.nan   # já winsorizada: max = p99 do treino
        return self

    def transform(self, X):
        X = X.copy()
        if 'sem_divida' in X:
            X['sem_divida'] = X['sem_divida'].astype(float)
        if self.regra == 'p99_zero' and 'sem_divida' in X:
            m = X['sem_divida'] == 1.0
            if 'fco_divida' in X:
                X.loc[m & X['fco_divida'].isna(), 'fco_divida'] = self.teto_fco_
            if 'divida_cp_share' in X:
                X.loc[m & X['divida_cp_share'].isna(), 'divida_cp_share'] = 0.0
        return X


def preprocessador(splines=False, usar_setor=True, regra=None, colunas_num=None):
    """(setores raros -> 'Outros setores') -> winsor -> regra do sem-dívida -> (imputação pela mediana + padronização) + flag + dummies de setor.

    splines=True troca as colunas de cfg.SPLINE_COLS por uma base de splines cúbicos (não linearidade da alavancagem
    e do tamanho, notebook 03). Os dummies de setor ficam completos (sem categoria de referência): a regularização L2
    resolve a colinearidade; a inferência (`logit_agrupado`) tira a categoria de referência."""
    num = list(colunas_num or cfg.FEATURES_NUM)
    macro = list(cfg.MACRO_COLS)
    spl = [c for c in cfg.SPLINE_COLS if c in num] if splines else []
    lineares = [c for c in num if c not in spl] + macro
    blocos = [('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('esc', StandardScaler())]), lineares)]
    if spl:
        blocos.append(('spl', Pipeline([('imp', SimpleImputer(strategy='median')),
                                        ('spl', SplineTransformer(n_knots=4, degree=3, knots='quantile',
                                                                  extrapolation='constant', include_bias=False)),
                                        ('esc', StandardScaler())]), spl))
    blocos.append(('flag', 'passthrough', list(cfg.FEATURES_FLAG)))
    if usar_setor:
        blocos.append(('setor', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [cfg.FEATURE_SETOR]))
    ct = ColumnTransformer(blocos, verbose_feature_names_out=False).set_output(transform='pandas')
    etapas = [('setores', AgruparSetores())] if usar_setor else []
    return Pipeline(etapas + [('winsor', Winsorizar(cols=num)),
                              ('sem_divida', RegraSemDivida(regra or cfg.REGRA_SEM_DIVIDA)),
                              ('ct', ct)])


# ---------------------------------------------------------------- modelos
def pipeline_z():
    """Baseline: Z'' de Altman (emergentes) recalibrado por uma logística de 1 variável.
    A ordenação (AUC-PR e AUC-ROC) é a do próprio Z''; a recalibração só serve para ter probabilidades (Brier)."""
    ct = ColumnTransformer([('z', Pipeline([('imp', SimpleImputer(strategy='median')), ('esc', StandardScaler())]),
                             ['z_score'])]).set_output(transform='pandas')
    return Pipeline([('winsor', Winsorizar(cols=['z_score'])), ('ct', ct),
                     ('clf', LogisticRegression(C=1e6, max_iter=1000))])


def pipeline_logit(C=1.0, class_weight='balanced', splines=False, usar_setor=True, regra=None):
    """Regressão logística com regularização L2. class_weight='balanced' melhora o recall, mas infla as
    probabilidades (leia-as como risco relativo); class_weight=None mantém a calibração."""
    return Pipeline([('prep', preprocessador(splines, usar_setor, regra)),
                     ('clf', LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=cfg.SEED))])


def pipeline_gb(usar_setor=True, regra=None):
    """Desafiante não linear (gradient boosting do scikit-learn), raso e regularizado de propósito: com poucas dezenas
    de eventos, árvores profundas decoram o treino. Hiperparâmetros FIXOS (não foram tunados); a troca por XGBoost
    é direta, basta substituir o classificador."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                         l2_regularization=1.0, class_weight='balanced', random_state=cfg.SEED)
    return Pipeline([('prep', preprocessador(False, usar_setor, regra)), ('clf', clf)])


def pipeline_rf(usar_setor=True, regra=None):
    """Random forest (o ML do plano inicial), também raso e com hiperparâmetros FIXOS: folhas grandes (poucos eventos)
    e pesos balanceados por árvore. Entra como segundo desafiante, sem seleção de modelo pelo teste."""
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=15, max_features='sqrt', class_weight='balanced_subsample',
                                 n_jobs=-1, random_state=cfg.SEED)
    return Pipeline([('prep', preprocessador(False, usar_setor, regra)), ('clf', clf)])


def prever(modelo, df):
    return modelo.predict_proba(df[colunas_entrada()])[:, 1]


def ajustar(modelo, df):
    modelo.fit(df[colunas_entrada()], df['Y'])
    return modelo


# ---------------------------------------------------------------- validação temporal expandida
def dobras_temporais(treino, anos_val=None, gap=None, min_eventos=None):
    """Dobras (idx_treino, idx_validacao, ano) POSICIONAIS sobre `treino`.

    A dobra do ano v treina com t <= v − gap − 1: o alvo de t olha t+1 e t+2, então com gap = 2 nenhum rótulo de treino
    usa informação dos anos da validação. Dobras com poucos eventos no treino são descartadas (e listadas em `.descartadas`)."""
    anos_val = anos_val or cfg.ANOS_VAL_CV
    gap = cfg.GAP_ANOS if gap is None else gap
    min_eventos = cfg.MIN_EVENTOS_DOBRA if min_eventos is None else min_eventos
    ano, y = treino['ano'].to_numpy(), treino['Y'].to_numpy()
    dobras, descartadas = [], []
    for v in anos_val:
        tr = np.flatnonzero(ano <= v - gap - 1)
        va = np.flatnonzero(ano == v)
        if y[tr].sum() < min_eventos or y[va].sum() == 0 or len(va) == 0:
            descartadas.append((v, int(y[tr].sum()), int(y[va].sum())))
            continue
        dobras.append((tr, va, v))
    return dobras, descartadas


def previsoes_fora_da_amostra(fabrica, treino, dobras):
    """Previsão de cada linha de validação de cada dobra, com o modelo ajustado só no treino da dobra.
    Devolve uma Series alinhada a `treino` (NaN nas linhas que nunca foram validadas)."""
    oof = pd.Series(np.nan, index=treino.index)
    for tr, va, _ in dobras:
        m = ajustar(fabrica(), treino.iloc[tr])
        oof.iloc[va] = prever(m, treino.iloc[va])
    return oof


def selecionar_c(treino, fabrica, grade=None, dobras=None):
    """AUC-PR agregada nas dobras (previsões empilhadas) para cada C, e escolha de C (cfg.REGRA_C):
    'max' = o de maior AUC-PR; '1ep' = o menor C (mais regularizado) a até 1 erro-padrão do melhor. A regra de 1 EP é a
    escolha clássica por parcimônia, mas com uma métrica de ranking a curva é plana e ela tende a ir para a ponta da
    grade (veja o aviso). `fabrica(C)` devolve um Pipeline novo.
    Devolve (tabela, C escolhido, previsões fora da amostra do C escolhido)."""
    grade = grade or cfg.GRADE_C
    dobras = dobras or dobras_temporais(treino)[0]
    if not dobras:
        raise ValueError('Nenhuma dobra válida: reveja cfg.ANOS_VAL_CV, cfg.GAP_ANOS e cfg.MIN_EVENTOS_DOBRA.')
    y = treino['Y']
    linhas, oofs = [], {}
    for C in grade:
        oof = previsoes_fora_da_amostra(lambda C=C: fabrica(C), treino, dobras)
        oofs[C] = oof
        m = oof.notna()
        met = avaliacao.metricas(y[m], oof[m])
        por_dobra = [avaliacao.metricas(y.iloc[va], oof.iloc[va])['AUC-PR'] for _, va, _ in dobras]
        linhas.append({'C': C, 'AUC-PR': met['AUC-PR'], 'AUC-ROC': met['AUC-ROC'], 'Brier': met['Brier'],
                       'AUC-PR por dobra (média)': np.nanmean(por_dobra)})
    tab = pd.DataFrame(linhas)
    melhor = tab.loc[tab['AUC-PR'].idxmax()]
    m = oofs[melhor['C']].notna()
    ep = avaliacao.se_auc_pr(y[m], oofs[melhor['C']][m], treino.loc[m, 'ticker'])
    c_esc = float(tab.loc[tab['AUC-PR'] >= melhor['AUC-PR'] - ep, 'C'].min()) if cfg.REGRA_C == '1ep' else float(melhor['C'])
    if c_esc == min(grade) or (c_esc == max(grade) and c_esc < 100):
        warnings.warn(f'C escolhido ({c_esc}) está na borda da grade: estenda cfg.GRADE_C nessa direção.')
    # C >= 100 na ponta alta = praticamente sem regularização; estender a grade não muda nada (curva plana)
    tab['melhor'] = tab['C'] == melhor['C']
    tab['escolhido'] = tab['C'] == c_esc
    tab.attrs['ep'] = ep
    return tab, c_esc, oofs[c_esc]


def ajuste_fixo(fabrica, treino, dobras=None):
    """Para modelos sem hiperparâmetro a escolher (Z'', gradient boosting): ajusta no treino e escolhe o limiar nas
    previsões fora da amostra das mesmas dobras temporais."""
    dobras = dobras or dobras_temporais(treino)[0]
    oof = previsoes_fora_da_amostra(fabrica, treino, dobras)
    m = oof.notna()
    return {'modelo': ajustar(fabrica(), treino), 'oof': oof, 'limiar': avaliacao.escolher_limiar(treino.loc[m, 'Y'], oof[m])}


def ajuste_completo(amostra, class_weight='balanced', splines=False, usar_setor=True, regra=None, grade=None):
    """Seleciona C por validação temporal no treino, ajusta no treino todo e escolhe o limiar nas previsões fora da
    amostra. O teste NÃO é tocado. Devolve um dicionário com modelo, C, tabela de C, limiar e previsões fora da amostra."""
    treino = amostra[amostra['split'] == 'treino'].reset_index(drop=True)
    fabrica = lambda C: pipeline_logit(C, class_weight, splines, usar_setor, regra)
    dobras, descartadas = dobras_temporais(treino)
    tab, c_esc, oof = selecionar_c(treino, fabrica, grade, dobras)
    modelo = ajustar(fabrica(c_esc), treino)
    m = oof.notna()
    return {'modelo': modelo, 'C': c_esc, 'tabela_C': tab, 'oof': oof, 'treino': treino,
            'limiar': avaliacao.escolher_limiar(treino.loc[m, 'Y'], oof[m]),
            'dobras': [(int(v), len(tr), len(va)) for tr, va, v in dobras], 'dobras_descartadas': descartadas}


# ---------------------------------------------------------------- inferência: logit sem penalização, erro-padrão por empresa
def _irls(X, y, max_iter=100, tol=1e-9):
    """Máxima verossimilhança da logística por mínimos quadrados reponderados (Newton-Raphson)."""
    b = np.zeros(X.shape[1])
    for it in range(max_iter):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1])
        passo = np.linalg.solve(H, X.T @ (y - p))
        b = b + passo
        if np.max(np.abs(passo)) < tol:
            return b, True
    return b, False


def _colunas_com_separacao(X, y):
    """Colunas 0/1 que, quando iguais a 1, têm zero evento (ou só eventos): causa típica de não convergência."""
    y = np.asarray(y)
    ruins = []
    for c in X.columns:
        v = X[c].to_numpy()
        if set(np.unique(v)) <= {0.0, 1.0} and (v == 1).any():
            if y[v == 1].sum() in (0, (v == 1).sum()):
                ruins.append(c)
    return ruins


def logit_agrupado(X, y, grupos):
    """Logit sem penalização (com constante) e erro-padrão robusto agrupado por empresa (sanduíche, com a correção
    de pequenas amostras G/(G−1)·(N−1)/(N−K), a mesma do Stata e do statsmodels com cov_type='cluster').
    O painel repete a mesma empresa em vários anos, então o erro-padrão comum seria otimista.
    `X`: DataFrame já pré-processado, SEM constante. Devolve DataFrame com coeficiente, EP, z, p e razão de chances."""
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    nomes = ['const'] + list(X.columns)
    y = np.asarray(y, dtype=float)
    b, ok = _irls(Xc, y)
    if not ok:
        ruins = _colunas_com_separacao(X, y)
        raise RuntimeError('A logística sem penalização não convergiu (separação perfeita). '
                           + (f'Colunas sem variação no alvo quando = 1: {ruins}. Junte categorias de setor ou retire a flag.'
                              if ruins else 'Nenhuma coluna 0/1 isolada explica; verifique variáveis quase colineares.'))
    p = 1 / (1 + np.exp(-np.clip(Xc @ b, -30, 30)))
    pao = np.linalg.inv(Xc.T @ (Xc * (p * (1 - p))[:, None]))
    escore = Xc * (y - p)[:, None]
    g = pd.factorize(np.asarray(grupos))[0]
    soma = np.zeros((g.max() + 1, Xc.shape[1]))
    np.add.at(soma, g, escore)
    G, N, K = soma.shape[0], len(y), Xc.shape[1]
    V = (G / (G - 1)) * ((N - 1) / (N - K)) * pao @ (soma.T @ soma) @ pao
    ep = np.sqrt(np.diag(V))
    z = b / ep
    lo, hi = cfg.IC
    zc = stats.norm.ppf(hi)
    out = pd.DataFrame({'coef': b, 'ep': ep, 'z': z, 'p': 2 * stats.norm.sf(np.abs(z)),
                        'odds_ratio': np.exp(b), 'or_inf': np.exp(b - zc * ep), 'or_sup': np.exp(b + zc * ep)}, index=nomes)
    out.attrs.update({'N': N, 'empresas': G, 'convergiu': ok})
    return out


def _agrupar_setores_raros(setor, y, minimo=None):
    """Setores com menos de `minimo` eventos no treino (cfg.MIN_EVENTOS_SETOR) viram 'Outros (poucos eventos)'.
    Se, juntos, ainda não tiverem nenhum evento, entram no setor mais frequente (a categoria de referência):
    uma categoria sem eventos faz a logística sem penalização divergir (separação perfeita)."""
    minimo = cfg.MIN_EVENTOS_SETOR if minimo is None else minimo
    ev = y.groupby(setor).sum()
    raros = ev.index[ev < minimo]
    if len(raros) == 0:
        return setor
    destino = 'Outros (poucos eventos)' if ev[raros].sum() >= 1 else setor.value_counts().idxmax()
    return setor.where(~setor.isin(raros), destino)


def matriz_inferencia(treino, regra='nenhuma', usar_setor=True):
    """Pré-processa o treino (winsor, padronização) e tira o setor de referência (o mais frequente).
    Os coeficientes das contínuas são por +1 desvio-padrão da variável winsorizada.

    Usa regra='nenhuma' (mediana) em vez de 'p99_zero' DE PROPÓSITO: com um constante + flag, qualquer constante dá o
    mesmo ajuste sem penalização (a flag absorve o nível), mas só a mediana deixa a flag legível ("sem dívida" contra uma
    empresa com dívida típica). Com o p99, o coeficiente de `sem_divida` só desfaz o valor imputado (razão de chances
    sem sentido, na casa das centenas).

    Setores com poucos eventos são agrupados (`_agrupar_setores_raros`); isso vale só para a inferência."""
    d = treino[colunas_entrada()].copy()
    if usar_setor:
        d[cfg.FEATURE_SETOR] = _agrupar_setores_raros(d[cfg.FEATURE_SETOR], treino['Y'])
    prep = preprocessador(False, usar_setor, regra)
    X = prep.fit_transform(d, treino['Y'])
    if usar_setor:
        dummies = [c for c in X.columns if c.startswith(cfg.FEATURE_SETOR + '_')]
        ref = X[dummies].sum().idxmax()
        X = X.drop(columns=ref)
        X.attrs['setor_referencia'] = ref.replace(cfg.FEATURE_SETOR + '_', '')
    return X


ROTULOS = {'alavancagem': 'Alavancagem (H1 +)', 'divida_cp_share': 'Dívida de curto prazo (H1 +)', 'roa': 'ROA (H2 −)',
           'liquidez': 'Liquidez (H3 −)', 'fco_divida': 'FCO / dívida (H4 −)', 'log_ativo': 'Tamanho (controle)',
           'sem_divida': 'Sem dívida (flag)'}


def tabela_hipoteses(coefs, nivel=0.10):
    """Compara o sinal estimado com o esperado (cfg.EXPLICATIVAS) para H1–H4.
    'confirma' = sinal esperado e p < nivel; 'sinal certo, não significativo'; 'contraria' = sinal oposto."""
    linhas = []
    for var, (h, sinal, _) in cfg.EXPLICATIVAS.items():
        if var not in coefs.index:
            continue
        r = coefs.loc[var]
        if sinal == '':
            ver = 'controle'
        else:
            certo = (r['coef'] > 0) == (sinal == '+')
            ver = ('confirma' if r['p'] < nivel else 'sinal certo, não significativo') if certo else \
                  ('contraria' if r['p'] < nivel else 'sinal oposto, não significativo')
        linhas.append({'variável': var, 'rótulo': ROTULOS.get(var, var), 'hipótese': h, 'sinal esperado': sinal or '—',
                       'sinal estimado': '+' if r['coef'] > 0 else '−', 'odds_ratio': r['odds_ratio'], 'or_inf': r['or_inf'],
                       'or_sup': r['or_sup'], 'p': r['p'], 'veredito': ver})
    return pd.DataFrame(linhas).set_index('variável')


# ---------------------------------------------------------------- regressão de referência: modelo linear de probabilidade
def mqo_agrupado(X, y, grupos):
    """Modelo linear de probabilidade (MQO com constante) e erro-padrão robusto agrupado por empresa. O coeficiente é a
    mudança na probabilidade de distress (em pontos percentuais, ×100) por +1 desvio-padrão. É a "regressão de referência":
    mais fácil de ler que a logística, mas com resíduos heterocedásticos por construção (Y é 0/1), por isso o EP robusto."""
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    y = np.asarray(y, dtype=float)
    pao = np.linalg.inv(Xc.T @ Xc)
    b = pao @ Xc.T @ y
    e = y - Xc @ b
    g = pd.factorize(np.asarray(grupos))[0]
    soma = np.zeros((g.max() + 1, Xc.shape[1]))
    np.add.at(soma, g, Xc * e[:, None])
    G, N, K = soma.shape[0], len(y), Xc.shape[1]
    V = (G / (G - 1)) * ((N - 1) / (N - K)) * pao @ (soma.T @ soma) @ pao
    ep = np.sqrt(np.diag(V))
    z = b / ep
    zc = stats.norm.ppf(cfg.IC[1])
    out = pd.DataFrame({'coef': b, 'ep': ep, 'z': z, 'p': 2 * stats.norm.sf(np.abs(z)),
                        'p.p.': 100 * b, 'pp_inf': 100 * (b - zc * ep), 'pp_sup': 100 * (b + zc * ep)},
                       index=['const'] + list(X.columns))
    out.attrs.update({'ajustados': Xc @ b, 'residuos': e, 'R2': 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()})
    return out


def vif_matriz(X):
    """Fator de inflação da variância de cada coluna contínua de `X` (matriz pré-processada de `matriz_inferencia`), calculado
    com a regressão da coluna nas demais. Referência usual: acima de 5 a 10 indica multicolinearidade preocupante."""
    A = X.to_numpy(dtype=float)
    out = {}
    for j, c in enumerate(X.columns):
        if c.startswith(cfg.FEATURE_SETOR + '_') or c in cfg.FEATURES_FLAG:
            continue
        outras = np.column_stack([np.ones(len(A)), np.delete(A, j, axis=1)])
        b, *_ = np.linalg.lstsq(outras, A[:, j], rcond=None)
        r2 = 1 - ((A[:, j] - outras @ b) ** 2).sum() / ((A[:, j] - A[:, j].mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name='VIF')


def ajustados_logit(X, coefs):
    """Probabilidades ajustadas da logística sem penalização (saída de `logit_agrupado`)."""
    eta = coefs.loc['const', 'coef'] + X.to_numpy(dtype=float) @ coefs.drop('const')['coef'].to_numpy()
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def residuos_binados(y, p, bins=10):
    """Resíduo médio (y − p) por faixa de valor ajustado, com ±2 erros-padrão. Se o modelo está bem especificado, as faixas
    ficam em torno de zero (dentro da banda). Usa quantis dos ajustados, então cada faixa tem ~o mesmo nº de observações."""
    d = pd.DataFrame({'y': np.asarray(y, dtype=float), 'p': np.asarray(p, dtype=float)})
    d['faixa'] = pd.qcut(d['p'], bins, duplicates='drop')
    g = d.groupby('faixa', observed=True)
    out = pd.DataFrame({'ajustado': g['p'].mean(), 'resíduo': g.apply(lambda x: (x['y'] - x['p']).mean()),
                        'n': g.size()})
    out['banda'] = 2 * g.apply(lambda x: (x['y'] - x['p']).std(ddof=1)).to_numpy() / np.sqrt(out['n'].to_numpy())
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- importância por permutação (comparar regressão e ML)
def importancia_permutacao(fabrica, treino, dobras, n_rep=20, seed=None):
    """Importância por permutação, SÓ com dados de treino: em cada dobra temporal ajusta no passado, embaralha uma variável
    por vez na validação e mede a queda da AUC-PR e da AUC-ROC. Média das dobras e das repetições. Variáveis em grupos:
    cada coluna de entrada (o setor conta como uma só). Queda grande = o modelo depende da variável; ~0 = ignora."""
    rng = np.random.default_rng(cfg.SEED if seed is None else seed)
    cols = list(cfg.FEATURES_NUM) + list(cfg.FEATURES_FLAG) + [cfg.FEATURE_SETOR]   # sem o z_score, que só o Z'' usa
    linhas = []
    for tr, va, ano in dobras:
        m = ajustar(fabrica(), treino.iloc[tr])
        val = treino.iloc[va].reset_index(drop=True)
        y = val['Y'].to_numpy()
        base = avaliacao.metricas(y, prever(m, val))
        for c in cols:
            for _ in range(n_rep):
                v = val.copy()
                v[c] = rng.permutation(v[c].to_numpy())
                mt = avaliacao.metricas(y, prever(m, v))
                linhas.append({'variável': c, 'dobra': ano, 'queda AUC-PR': base['AUC-PR'] - mt['AUC-PR'],
                               'queda AUC-ROC': base['AUC-ROC'] - mt['AUC-ROC']})
    d = pd.DataFrame(linhas)
    por_dobra = d.groupby(['variável', 'dobra']).mean().groupby('variável').mean()
    return por_dobra.sort_values('queda AUC-ROC', ascending=False)


# ---------------------------------------------------------------- cenários de robustez
def avaliar_cenario(amostra, nome, **kw_logit):
    """Roda o pipeline inteiro (escolha de C no treino -> ajuste -> avaliação no TESTE) para uma amostra alternativa e
    compara a logística com o Z''. Use só na robustez, depois de congelar o modelo principal."""
    r = ajuste_completo(amostra, **kw_logit)
    tr, te = r['treino'], amostra[amostra['split'] == 'teste']
    z = ajustar(pipeline_z(), tr)
    ml, mz = avaliacao.metricas(te['Y'], prever(r['modelo'], te)), avaliacao.metricas(te['Y'], prever(z, te))
    return {'cenário': nome, 'n treino': len(tr), 'eventos treino': int(tr['Y'].sum()), 'n teste': ml['n'],
            'eventos teste': ml['eventos'], 'C': r['C'], 'AUC-PR logística': ml['AUC-PR'], "AUC-PR Z''": mz['AUC-PR'],
            'AUC-ROC logística': ml['AUC-ROC'], "AUC-ROC Z''": mz['AUC-ROC'], 'sem informação (AUC-PR)': ml['taxa']}