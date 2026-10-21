"""Constantes do projeto. Tudo que é escolha de projeto fica aqui, num lugar só."""
from pathlib import Path

# ---------------------------------------------------------------- caminhos
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / 'data' / 'raw'
DATA_PROCESSED = RAIZ / 'data' / 'processed'
FIGURES = RAIZ / 'figures'
FONTS = RAIZ / 'src' / 'fonts'

ARQ_ECONOMATICA = [
    DATA_RAW / 'economatica' / 'Economatica-8900590473-20261020215038.csv',
    DATA_RAW / 'economatica' / 'extra.csv',
    DATA_RAW / 'economatica' / 'juros_pagaos_&_capex.csv',
]
ARQ_CAD_CVM = DATA_RAW / 'cvm' / 'cad_cia_aberta.csv'
DIR_FCA_CVM = DATA_RAW / 'cvm' / 'fca'

# ---------------------------------------------------------------- período
ANO_INI = 2010
ANO_FIM = 2026          # último ano exportado do Economatica
ANO_T_MAX = 2023        # o alvo de t usa t+1 e t+2 -> último t com alvo observável

# Divisão temporal (anos de t, inclusive). O gap evita vazamento: o alvo de t usa t+1 e t+2,
# então o último alvo do treino (t=2018) olha 2019-2020, e o teste começa em 2021.
# Decisão da dupla (21/10/2026): treino de ~10 anos e teste com os anos mais recentes.
SPLITS = {
    'treino': (2010, 2018),
    'validacao': (2019, 2020),
    'teste': (2021, 2023),
}

# ---------------------------------------------------------------- definições de alvo (Etapa B)
# nome: (numerador da cobertura, nº de anos seguidos em distress depois de t)
DEFINICOES_ALVO = {
    'A': ('ebit', 2),     # atual
    'B': ('ebitda', 2),
    'C': ('ebit', 3),
    'D': ('ebitda', 3),
}
ALVO_ATUAL = 'A'
ULTIMO_ANO_DADOS = 2025          # último 4º trimestre disponível
PREVALENCIA_REF = (0.15, 0.20)   # faixa de referência da literatura para empresas "zumbi" (BIS; Granzotto et al., 2025)
# Parâmetros previstos na logística: alavancagem, divida_cp_share, roa, liquidez, fco_divida, sem_divida,
# log_ativo, Selic (8) + 5 dummies de macrossetor (6 macrossetores) = 13, sem contar o intercepto
N_PARAMETROS_PREVISTOS = 13
ARQ_EVENTOS = DATA_RAW / 'eventos' / 'eventos_credito.csv'

# ---------------------------------------------------------------- tratamento
WINSOR = (0.01, 0.99)   # percentis calculados só no treino

# ---------------------------------------------------------------- colunas
# Prefixo do nome original do Economatica (antes do primeiro "|") -> nome curto.
# Atenção: "ResLuc" no Economatica é RESERVAS DE LUCRO (conta de balanço), não resultado líquido.
RENOMEAR = {
    'DesFin': 'desp_financeira',            # no exercício (acumulado no ano)
    'Ebitda/Fin': 'ebitda_desp_fin',
    'Ativo Tot': 'ativo_total',
    'Div Tt Bruta': 'divida_bruta',
    'Div Tt Lq': 'divida_liquida',
    'Patrim Liq': 'patrimonio_liquido',
    'LiqCor': 'liquidez_corrente',
    'CaixaEEqCx': 'caixa',
    'TotEmFiCP': 'emprestimos_cp',
    'TotEmFiLP': 'emprestimos_lp',
    'ResLuc': 'reservas_lucro',
    'AtvCir': 'ativo_circulante',
    'PasCir': 'passivo_circulante',
    'Receita': 'receita',                   # no exercício (acumulado no ano)
    'Valor Mercado': 'valor_mercado',
    'Pas+PL': 'passivo_mais_pl',
    'EBIT': 'ebit',                         # 12 meses
    'EBITDA': 'ebitda',                     # 12 meses
    'Lucro Liquido': 'lucro_liquido_12m',   # 12 meses
    'CxOper': 'caixa_operacional_12m',      # 12 meses, em R$ (não em milhares)
    'Div Moed Estr': 'divida_moeda_estrangeira',
    'JurosPagos': 'juros_pagos',            # veio vazio, não usado
    'Capex (exc inv)': 'capex_12m',         # veio vazio, não usado
}

# Colunas que o Economatica exportou em R$ e não em milhares (padronizamos para milhares)
EM_REAIS = ['caixa_operacional_12m', 'juros_pagos']

# Colunas que vieram 100% vazias e não serão usadas
DESCARTAR_VAZIAS = ['juros_pagos', 'capex_12m']

# ---------------------------------------------------------------- ligação com a CVM
# Tickers que não casam automaticamente com o FCA da CVM (nem pelo código exato,
# nem pela raiz de 4 letras). Mapeados à mão pelo nome no cadastro da CVM.
# 'conferir' = palpite plausível que a dupla precisa confirmar no Economatica.
MAPA_MANUAL_CNPJ = {
    'CSNA3': ('33.042.730/0001-04', 'ok', 'CIA SIDERURGICA NACIONAL'),
    'CMIN3': ('08.902.291/0001-15', 'ok', 'CSN MINERAÇÃO'),
    'MBRF3': ('03.853.896/0001-40', 'ok', 'MARFRIG (MBRF Global Foods)'),
    'NORD3': ('60.884.319/0001-59', 'ok', 'NORDON'),
    'HAGA4': ('30.540.991/0001-66', 'ok', 'HAGA'),
    'AMAR3': ('61.189.288/0001-89', 'ok', 'MARISA LOJAS'),
    'LUXM4': ('92.660.570/0001-26', 'ok', 'TREVISA INVESTIMENTOS'),
    'EQPA3': ('04.895.728/0001-80', 'ok', 'EQUATORIAL PARÁ'),
    'PLAS3': ('51.928.174/0001-50', 'ok', 'PLASCAR'),
    'CTAX3': ('04.032.433/0001-80', 'ok', 'CONTAX PARTICIPAÇÕES'),
    # confirmados pela dupla em 21/10/2026
    'JBSS32': ('02.916.265/0001-60', 'ok', 'JBS S.A. (BDR JBSS32 substituiu JBSS3; dupla listagem com NY)'),
    'COMR3': ('25.369.840/0001-57', 'ok', 'COMERC ENERGIA (adquirida pela Vibra)'),
    'IGSN3': ('08.159.965/0001-33', 'ok', 'IGUA SANEAMENTO (registrada na CVM, nunca listou)'),
    'EGGY3': ('81.616.807/0001-55', 'conferir', 'GRANJA FARIA'),
    # LLBI3: em recuperação judicial, registro cancelado de ofício pela CVM; não achado no cadastro
}

# Financeiras que sobraram no Screening (setor CVM). Decisão da dupla: excluir todas.
EXCLUIR_FINANCEIRAS = ['B1003', 'B3SA3', 'BRBI11', 'CXSE3', 'HBTS5', 'ITSA4', 'QUAL3', 'WIZC3']

# Setor PROVISÓRIO: agrupamento do SETOR_ATIV da CVM em macrossetores (inspirado nos setores
# econômicos da B3). Será trocado pelo Subsetor Bovespa quando a dupla exportar do Economatica.
# "Emp. Adm. Part. - X" (holding) vai para o mesmo grupo de X.
SETOR_MACRO = {
    'Energia Elétrica': 'Utilidade pública',
    'Saneamento, Serv. Água e Gás': 'Utilidade pública',
    'Construção Civil, Mat. Constr. e Decoração': 'Construção e imobiliário',
    'Const. Civil, Mat. Const. e Decoração': 'Construção e imobiliário',
    'Comércio (Atacado e Varejo)': 'Consumo cíclico',
    'Têxtil e Vestuário': 'Consumo cíclico',
    'Hospedagem e Turismo': 'Consumo cíclico',
    'Brinquedos e Lazer': 'Consumo cíclico',
    'Educação': 'Consumo cíclico',
    'Alimentos': 'Consumo não cíclico',
    'Bebidas e Fumo': 'Consumo não cíclico',
    'Agricultura (Açúcar, Álcool e Cana)': 'Consumo não cíclico',
    'Serviços Médicos': 'Saúde',
    'Serviços médicos': 'Saúde',
    'Farmacêutico e Higiene': 'Saúde',
    'Máquinas, Equipamentos, Veículos e Peças': 'Bens industriais',
    'Máqs., Equip., Veíc. e Peças': 'Bens industriais',
    'Serviços Transporte e Logística': 'Bens industriais',
    'Metalurgia e Siderurgia': 'Materiais básicos',
    'Petroquímicos e Borracha': 'Materiais básicos',
    'Papel e Celulose': 'Materiais básicos',
    'Extração Mineral': 'Materiais básicos',
    'Embalagens': 'Materiais básicos',
    'Petróleo e Gás': 'Petróleo e gás',
    'Comunicação e Informática': 'Comunicações e TI',
    'Telecomunicações': 'Comunicações e TI',
    'Sem Setor Principal': 'Holdings e outros',
}

# Explicativas (índices). H5 (dívida em moeda estrangeira) foi abandonada: o Economatica só trouxe 2010.
EXPLICATIVAS = {
    'alavancagem': ('H1', '+', 'dívida bruta / ativo total'),
    'divida_cp_share': ('H1', '+', 'empréstimos CP / (CP + LP)'),
    'roa': ('H2', '−', 'lucro líquido 12m / ativo total'),
    'liquidez': ('H3', '−', 'caixa / passivo circulante'),
    'fco_divida': ('H4', '−', 'caixa operacional 12m / dívida bruta'),
    'log_ativo': ('controle', '', 'log do ativo total (R$ mil)'),
}

# SETOR_ATIV da CVM considerados financeiros (bancos, seguradoras, crédito, securitização, bolsa)
SETORES_FINANCEIROS = [
    'Bancos',
    'Seguradoras e Corretoras',
    'Intermediação Financeira',
    'Arrendamento Mercantil',
    'Crédito Imobiliário',
    'Securitização de Recebíveis',
    'Bolsas de Valores/Mercadorias e Futuros',
    'Emp. Adm. Part. - Intermediação Financeira',
    'Emp. Adm. Part. - Seguradoras e Corretoras',
    'Emp. Adm. Part. - Crédito Imobiliário',
    'Emp. Adm. Part. - Securitização de Recebíveis',
    'Emp. Adm. Part. - Arrendamento Mercantil',
    'Emp. Adm. Part. - Bancos',
]

# ---------------------------------------------------------------- identidade visual
COR_DESTAQUE = '#FF4713'
CINZA = '#C4C4C4'
CINZA_2 = '#D0D0D0'
CINZA_GRID = '#E7E7E7'
COR_TITULO = '#111111'
COR_SUBTITULO = '#595959'
COR_EIXO = '#2B2B2B'
FONTE_RODAPE = 'Fonte: Economatica e CVM; elaboração própria'
DPI = 200
TAMANHO_FIG = (12, 6.75)   # 16:9
