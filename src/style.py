"""Identidade visual dos gráficos (igual aos slides). Chame `aplicar()` uma vez no notebook."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from . import config as cfg

FONTE = None


def _registrar_fonte():
    """Registra a Montserrat que está em src/fonts/. Se faltar, cai para Inter ou DejaVu Sans."""
    for f in cfg.FONTS.glob('*.ttf'):
        font_manager.fontManager.addfont(str(f))
    disponiveis = {f.name for f in font_manager.fontManager.ttflist}
    for nome in ['Montserrat', 'Inter', 'DejaVu Sans']:
        if nome in disponiveis:
            if nome != 'Montserrat':
                print(f'Aviso: Montserrat não encontrada, usando {nome}.')
            return nome


def aplicar():
    global FONTE
    FONTE = _registrar_fonte()
    mpl.rcParams.update({
        'font.family': FONTE,
        'font.size': 11,
        'figure.figsize': cfg.TAMANHO_FIG,
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': cfg.DPI,
        'axes.facecolor': 'white',
        'axes.edgecolor': cfg.CINZA,
        'axes.labelcolor': cfg.COR_SUBTITULO,
        'axes.titlelocation': 'left',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.prop_cycle': mpl.cycler(color=[cfg.COR_DESTAQUE, cfg.CINZA, '#8C8C8C', cfg.CINZA_2]),
        'grid.color': cfg.CINZA_GRID,
        'grid.linewidth': 0.6,
        'xtick.color': cfg.COR_SUBTITULO,
        'ytick.color': cfg.COR_SUBTITULO,
        'xtick.labelcolor': cfg.COR_EIXO,
        'ytick.labelcolor': cfg.COR_EIXO,
        'legend.frameon': False,
        'axes.formatter.use_locale': False,
    })


# ---------------------------------------------------------------- números em pt-BR
def num_br(x, casas=0):
    """1234567.891 -> '1.234.567,89' (casas=2)."""
    s = f'{x:,.{casas}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def pct_br(x, casas=1):
    """0.1234 -> '12,3%'."""
    return num_br(100 * x, casas) + '%'


def formatador(casas=0, pct=False):
    """Formatador de eixo com vírgula decimal."""
    if pct:
        return FuncFormatter(lambda x, _: pct_br(x, casas))
    return FuncFormatter(lambda x, _: num_br(x, casas))


# ---------------------------------------------------------------- mapas de cor
CMAP_SEQ = LinearSegmentedColormap.from_list('branco_laranja', ['#FFFFFF', cfg.COR_DESTAQUE])
CMAP_DIV = LinearSegmentedColormap.from_list('cinza_laranja', ['#8C8C8C', '#FFFFFF', cfg.COR_DESTAQUE])


# ---------------------------------------------------------------- texto
def titulos(fig, titulo, subtitulo=None):
    """Título (afirmação, negrito) e subtítulo (o que é medido e a unidade), alinhados à esquerda."""
    fig.text(0.01, 0.97, titulo, ha='left', va='top', fontsize=17, fontweight='bold', color=cfg.COR_TITULO)
    if subtitulo:
        fig.text(0.01, 0.905, subtitulo, ha='left', va='top', fontsize=11.5, color=cfg.COR_SUBTITULO)
    fig.text(0.01, 0.015, cfg.FONTE_RODAPE, ha='left', va='bottom', fontsize=8.5, color=cfg.COR_SUBTITULO)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.97)


def rotular_barras(ax, barras, casas=0, pct=False, cor=None, dentro=False):
    """Rótulo de dado direto na barra (em vez de eixo carregado)."""
    for b in barras:
        v = b.get_height()
        if v != v:   # NaN
            continue
        txt = pct_br(v, casas) if pct else num_br(v, casas)
        y = v / 2 if dentro else v
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, y), xytext=(0, 0 if dentro else 3),
                    textcoords='offset points', ha='center', va='center' if dentro else 'bottom',
                    fontsize=9, color=cor or cfg.COR_EIXO)


def salvar(fig, nome):
    """Exporta em figures/ como PNG 200 dpi, fundo branco."""
    cfg.FIGURES.mkdir(exist_ok=True)
    caminho = cfg.FIGURES / f'{nome}.png'
    fig.savefig(caminho, dpi=cfg.DPI, facecolor='white')
    return caminho
