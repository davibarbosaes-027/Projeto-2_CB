"""Painel SINTÉTICO com o esquema de data/processed/base_empresa_ano.parquet, só para testar o código da Fase 4
sem a base licenciada. Os números NÃO representam empresas reais e não devem aparecer em nenhum resultado.

Passa pelas funções reais de src/base.py (explicativas, Z'', divisão, winsorização), para checar que a camada de
modelagem encaixa nelas. O alvo em si é calculado depois, por src/alvo.py, como no notebook 04."""
import numpy as np
import pandas as pd

from src import base, config as cfg

SETORES = ['Utilidade pública', 'Construção e imobiliário', 'Consumo cíclico', 'Consumo não cíclico',
           'Saúde', 'Bens industriais', 'Materiais básicos', 'Petróleo e gás']


def gerar(n_empresas=300, seed=0):
    rng = np.random.default_rng(seed)
    choque_ano = {a: 0.0 for a in range(2010, 2027)}
    choque_ano.update({2014: -0.4, 2015: -0.8, 2016: -0.7, 2020: -0.6, 2022: -0.3, 2023: -0.4})
    linhas = []
    for i in range(n_empresas):
        tk = f'SIM{i:03d}3'
        ini = int(rng.choice(range(2010, 2019), p=[.6] + [.05] * 8)) if rng.random() < .8 else 2010
        setor = SETORES[rng.integers(len(SETORES))]
        log_ativo = rng.normal(15.3, 1.4)
        sem_divida = rng.random() < 0.09
        h = rng.normal(0, 1)
        for ano in range(ini, 2026):
            h = 0.72 * h + 0.69 * rng.normal() + choque_ano[ano] * 0.5
            at = np.exp(log_ativo + 0.03 * (ano - 2010) + rng.normal(0, .05))
            alav = 0 if sem_divida else float(np.clip(0.26 - 0.07 * h + rng.normal(0, .07), 0.01, 1.4))
            divida = alav * at
            cp_share = float(np.clip(0.30 - 0.05 * h + rng.normal(0, .15), 0.02, 0.98))
            roa = 0.025 + 0.035 * h + rng.normal(0, .025)
            ebit = at * (0.055 + 0.045 * h + rng.normal(0, .02))
            ebitda = ebit + at * (0.03 + rng.normal(0, .008))
            desp = divida * (0.115 + rng.normal(0, .02)) + (at * (0.03 + 0.02 * rng.random()) if sem_divida else 0)   # arrendamento / variação cambial
            if rng.random() < 0.04:
                desp = -abs(desp) * .1          # contaminação cambial: despesa <= 0 -> cobertura NaN
            caixa = at * np.clip(0.08 + 0.03 * h + rng.normal(0, .03), 0.002, None)
            pc = at * np.clip(0.18 - 0.02 * h + rng.normal(0, .05), 0.03, None)
            linhas.append(dict(
                ticker=tk, ano=ano, setor=setor, ativo_total=at, divida_bruta=divida,
                emprestimos_cp=divida * cp_share, emprestimos_lp=divida * (1 - cp_share), caixa=caixa,
                passivo_circulante=pc, ativo_circulante=pc * np.clip(1.3 + 0.2 * h + rng.normal(0, .3), 0.3, None),
                patrimonio_liquido=at * np.clip(0.4 + 0.08 * h + rng.normal(0, .1), -0.2, 0.9),
                reservas_lucro=at * (0.08 + 0.05 * h + rng.normal(0, .06)), lucro_liquido_12m=at * roa,
                caixa_operacional_12m=at * (0.05 + 0.04 * h + rng.normal(0, .03)), ebit=ebit, ebitda=ebitda,
                desp_financeira=desp, receita=at * 0.8, ano_registro=ini + int(rng.integers(0, 3))))
    ea = pd.DataFrame(linhas)
    ea['pre_registro'] = ea['ano'] < ea['ano_registro']
    return processar(ea)


def processar(ea):
    """O que base.construir() faz depois da junção com a CVM (src/base.py)."""
    ea = base.adicionar_alvo(ea)
    ea = base.adicionar_explicativas(ea)
    ea = base.adicionar_zscore(ea)
    ea = base.adicionar_split(ea)
    ea = base.aplicar_winsor(ea, base.limites_winsor(ea, list(cfg.EXPLICATIVAS)))
    return ea.sort_values(['ticker', 'ano']).reset_index(drop=True)