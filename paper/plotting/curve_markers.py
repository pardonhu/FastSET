"""Presentation-only marker placement on existing measured SNR coordinates."""
import numpy as np
from matplotlib.lines import Line2D


def style_main(fig):
    styles = {
        'FastSET': ('#0072BD', 'o', '-', 0),
        'CNN': ('#D95319', 's', '--', 1),
        'E2E': ('#D95319', 's', '--', 1),
        'CNN est.': ('#EDB120', 'D', '-.', 2),
        'LS': ('#7E2F8E', '^', (0, (5, 2)), 0),
        'Perfect CSI': ('#77AC30', 'x', ':', 1),
    }
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_label() not in styles:
                continue
            color, marker, ls, offset = styles[line.get_label()]
            line.set_color(color); line.set_marker(marker); line.set_linestyle(ls)
            line.set_linewidth(2); line.set_markersize(4.5)
            line.set_markerfacecolor('white'); line.set_markeredgecolor(color)
            line.set_markeredgewidth(1)
            line.set_markevery([i for i,x in enumerate(line.get_xdata())
                                if (int(round(x))-4-offset) % 3 == 0])
        for text in ax.texts:
            if text.get_text() == 'Target: 2e-4':
                text.set_text(r'$2\times10^{-4}$')
                text.set_color('black')
    # Rebuild the existing legend handles from the updated line styles.
    for legend in fig.legends:
        lookup = {l.get_label():l for l in fig.axes[0].lines}
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            line = lookup.get(text.get_text())
            if line is not None:
                handle.update_from(line)
                handle.set_markevery(None)


RT_BER = {'four_city_ber', 'paris_P2_P3_ber', 'cross_city_ber'}
PAIRED_BER = {'environment_transition_performance', 'multipath_delay_ber', 'rt_delay_ber'}


def prepare_overlap(fig, name):
    if name not in RT_BER or getattr(fig, '_overlap_prepared', False):
        return
    fig._overlap_prepared = True
    for ax in fig.axes:
        lines = [line for line in ax.lines if not line.get_label().startswith('_')]
        assert len(lines) == 4, (name, len(lines))
        for offset, line in enumerate(lines):
            # Keep original zero-error downward triangles; ordinary markers show positive BER only.
            indices = [i for i,x in enumerate(line.get_xdata())
                       if (int(round(x))-4-offset)%4 == 0 and line.get_ydata()[i] > 1/26054400]
            line.set_markevery(indices)
            line.set_linewidth(2)
        for text in ax.texts:
            if 'epsilon' in text.get_text():
                text.set_text(r'$2\times10^{-4}$')
                text.set_color('black')


def lift_markers(fig, name):
    """Render markers above all curves, retaining every original curve vertex."""
    if name not in {'main_ber', 'speed_ber'} | RT_BER | PAIRED_BER or getattr(fig, '_markers_lifted', False):
        return
    fig._markers_lifted = True
    report = []
    axes = fig.axes if name == 'main_ber' or name in RT_BER else fig.axes[:1]
    for panel, ax in enumerate(axes):
        for line in list(ax.lines):
            if line.get_label().startswith('_'):
                continue
            x = np.asarray(line.get_xdata()); y = np.asarray(line.get_ydata())
            idx = line.get_markevery()
            assert isinstance(idx, list), (name, line.get_label(), idx)
            overlay = Line2D(x[idx], y[idx], linestyle='none',
                             marker=line.get_marker(), markersize=4.5,
                             markeredgewidth=1, markerfacecolor='white',
                             markeredgecolor=line.get_color(), color=line.get_color(),
                             zorder=10, label='_marker_overlay')
            ax.add_line(overlay)
            line.set_markevery([])
            report.append({'panel':panel,'label':line.get_label(),
                           'marker_snr':x[idx].tolist(),
                           'all_curve_vertices':np.column_stack([x,y]).tolist()})
    fig._marker_report = report
