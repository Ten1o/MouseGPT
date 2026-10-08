"""Four rubric dimensions and their exact per-frame index relationship."""
from panel_common import Panel, BLUE, ORANGE, GREEN


def make_panel():
    p = Panel('c', 'MouseGPT Benchmark Index', 98, 33)
    # The four named dimensions match panel d. P and R combine into F;
    # S and A modify the factual term multiplicatively.
    components = [
        (1.0, 'P', 'Factual\ncorrectness', None, BLUE),
        (26.0, 'R', 'Coverage', 'Required facts', BLUE),
        (51.0, 'S', 'Structure', 'Six semantic fields', GREEN),
        (76.0, 'A', 'Instructions', 'Task adherence', ORANGE),
    ]
    for x, symbol, first_line, second_line, color in components:
        p.text(x, 7.5, '$'+symbol+'$', size=10.5, color=color)
        p.text(x+5.5, 6.7, first_line, size=6.3, weight='bold', linespacing=1.1)
        if second_line:
            p.text(x+5.5, 10.0, second_line, size=6.0)
    # A shared bracket exposes the two ingredients of factual quality.
    p.line(2.0, 14.3, 47.0, 14.3, color='#9AB0CB', lw=.55)
    p.line(2.0, 13.5, 2.0, 14.3, color='#9AB0CB', lw=.55)
    p.line(47.0, 13.5, 47.0, 14.3, color='#9AB0CB', lw=.55)
    p.text(24.5, 15.5, r'$F=\frac{2PR}{P+R}$', size=9.5, ha='center', color=BLUE)
    p.text(73.0, 17.0, 'Components on a 0–1 scale', size=6.0, ha='center')
    p.rect(0,23.0,98,6.0,fill='#F1F4F2',edge='none',lw=0)
    p.fig.artists[-1].set_zorder(-10)
    p.text(49, 23.7, r'$I=100F\,\left(0.8+0.1S+0.1A\right)$', size=11.0, ha='center')
    p.text(49, 30.0, 'Per-frame index (0–100), then averaged', size=6.0, ha='center')
    p.save()


if __name__ == '__main__':
    make_panel()
