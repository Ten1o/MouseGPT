"""Graphical benchmark overview: reference stack, VLM bank, rubric scorecard."""
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

from panel_common import Panel, GREEN, BLUE, INK, DATA

EXAMPLES = DATA / 'examples'


def rounded(p, x, y, width, height, fill='white', edge='#96A398', lw=.5,
            radius=.8, zorder=1):
    patch = FancyBboxPatch(
        (x / p.w, 1 - (y + height) / p.h), width / p.w, height / p.h,
        transform=p.fig.transFigure,
        boxstyle=f'round,pad=0,rounding_size={radius / p.w}',
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    p.fig.add_artist(patch)


def circle(p, x, y, radius, fill='white', edge=GREEN, lw=.5, zorder=3):
    # Use figure inches so the physical circle remains circular.
    patch = Circle((x / 25.4, (p.h - y) / 25.4), radius / 25.4,
                   transform=p.fig.dpi_scale_trans, facecolor=fill,
                   edgecolor=edge, linewidth=lw, zorder=zorder)
    p.fig.add_artist(patch)


def tick(p, x, y, color=GREEN, lw=.8):
    p.line(x, y, x + .85, y + .8, color=color, lw=lw, zorder=6)
    p.line(x + .85, y + .8, x + 2.15, y - 1.0, color=color, lw=lw, zorder=6)


def main():
    p = Panel('a', 'Benchmark overview', 98, 31)
    rounded(p, 0, 4.7, 98, 25.7, fill='#F1F6F2', edge='none',
            lw=0, radius=2.5, zorder=-10)

    # The three illustrations are peer benchmark components, without connectors.

    # Expert-verified references: a two-camera image stack backed by a
    # reference sheet, not an enlarged duplicate of the example panel.
    p.text(11.8, 5.5, 'Expert-verified', ha='center', size=6.0, weight='bold')
    p.text(11.8, 8.15, 'ground-truth data', ha='center', size=6.0)
    rounded(p, 5.5, 11.25, 16.6, 11.4, fill='white', edge='#AAB6AD',
            lw=.45, radius=.45, zorder=1)
    for k, w in enumerate((12.2, 9.4, 11.1)):
        p.line(7, 12.4 + k * 1.3, 7 + w, 12.4 + k * 1.3,
               color='#BAC8BF', lw=.55, zorder=2)
    frame = '231117_G5_30min_00007684'
    for view, x in ((1, 1.2), (2, 11.9)):
        rounded(p, x, 15.0, 10.0, 9.9, fill='white', edge='#8D9990',
                lw=.45, radius=.3, zorder=2)
        ax = p.ax(x + .45, 15.45, 9.1, 8.9)
        ax.set_zorder(3)
        ax.imshow(Image.open(EXAMPLES / f'{frame}_view{view}.jpg'),
                  interpolation='none')
        ax.axis('off')
    circle(p, 20.75, 13.55, 2.0, fill='white', edge=GREEN, lw=.6, zorder=5)
    tick(p, 19.75, 13.6, color=GREEN, lw=.9)
    p.text(11.8, 27.0, '435 frames', ha='center', size=6.4, weight='bold')

    # A model-bank / chip miniature carries actual identities. The pins are
    # schematic and do not encode architecture, capacity, or model scores.
    p.text(49.7, 5.5, 'VLM Evaluations', ha='center', size=6.4, weight='bold')
    for y in (11.3, 14.4, 17.5, 20.6, 23.7):
        p.line(26.7, y, 28.2, y, color='#7C93AA', lw=.7, zorder=1)
        p.line(71.2, y, 72.7, y, color='#7C93AA', lw=.7, zorder=1)
    rounded(p, 28.0, 8.8, 43.4, 17.0, fill='#F7FAFD', edge='#6E88A1',
            lw=.6, radius=.9, zorder=2)
    p.line(49.7, 11.7, 49.7, 22.7, color='#D7E1EA', lw=.45, zorder=3)
    # Broad family-selection scope, not individual launch or snapshot dates.
    p.text(30.2, 9.3, 'mid-2024', size=6.0, color='#526B80', zorder=5)
    p.text(69.2, 23.2, 'late-2026', size=6.0, color='#526B80', ha='right', zorder=5)
    models = [
        (30.2, ['GPT-4o', 'GPT-4.1', 'GPT-5', 'GPT-5.6 Sol']),
        (51.7, ['GPT-5.6 Luna', 'GPT-6 Astra', 'GPT-6 Sol', 'GPT-6 Luna']),
    ]
    for x, names in models:
        for row, name in enumerate(names):
            p.text(x, 11.9 + row * 2.9, name, size=6.3, zorder=5)
    p.text(49.7, 27.0, '8 models · 14 settings', ha='center', size=6.2,
           weight='bold')

    # AI rubric scorecard: checks denote evaluated criteria, not fabricated
    # successful outcomes or numerical scores.
    p.text(86.6, 5.5, 'AI grading', ha='center', size=6.2, weight='bold')
    rounded(p, 76.3, 10.5, 20.6, 18.4, fill='white', edge='#9AA99F',
            lw=.5, radius=.7, zorder=1)
    # Short paired strokes denote candidate/reference comparison.
    for y in (12.0, 13.0):
        p.line(79.0, y, 84.8, y, color=BLUE, lw=.55, zorder=2)
        p.line(88.3, y, 94.1, y, color=GREEN, lw=.55, zorder=2)
    p.line(86.5, 11.8, 86.5, 13.2, color='#B7C1BA', lw=.45, zorder=2)
    for y, label in ((16.0, 'Facts'), (20.0, 'Structure'), (24.0, 'Instructions')):
        # Small outlined boxes represent rubric criteria, with no pass/fail
        # marks inside; the reference verification mark is distinct.
        p.rect(78.0, y + .2, 1.25, 1.25, fill='#EFF4EF', edge=GREEN, lw=.4)
        p.fig.artists[-1].set_zorder(3)
        p.text(80.5, y - .1, label, size=6.0, zorder=5)
    p.save()


if __name__ == '__main__':
    main()
