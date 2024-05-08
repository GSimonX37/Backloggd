import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.gridspec import GridSpec

from .step import step


def studies(trials: pd.DataFrame,
            title: str,
            path: str = None):
    """

    :param trials:
    :param title:
    :param path:
    :return:
    """

    sns.set_style('white')

    # Создаем объект фигуры.
    figure = plt.figure(
            layout='constrained',
            figsize=(20, 15)
    )

    # Определяем заголовок для фигуры.
    figure.suptitle(
            t=f'Исследование пространства гиперпараметров модели {title}',
            fontsize='x-large',
            y=1.025,
    )

    # Создаем сетку для размещения графиков.
    grid = GridSpec(
            figure=figure,
            nrows=2,
            ncols=1,
            hspace=0.05,
    )

    # Добавляем оси для графиков в сетку.
    figure.add_subplot(grid[0, 0])
    figure.add_subplot(grid[1, 0])

    order = (trials
             .groupby('job')['best']
             .max()
             .sort_values()
             .index
             .to_list()[::-1])
    trials['target'] = trials['job'] == order[0]

    sns.lineplot(
            data=trials,
            x='index',
            y='values',
            linewidth=1.5,
            style='job',
            hue='target',
            style_order=order,
            hue_order=[True, False],
            ax=figure.axes[0],
            palette=sns.color_palette('hls', 15)[8::5][::-1]
    )

    # Определяем стиль графика 1.
    # Определяем заголовок.
    figure.axes[0].set_title(
            label='Зависимость метрики F1-weighted от испытания',
            fontsize='large',
            y=1.035
    )

    figure.axes[0].set_xlabel('Номер испытания')
    figure.axes[0].set_ylabel('Метрика F1-weighted')
    # Определяем подписи значений для оси.
    s, m = step(trials['index'].max())
    figure.axes[0].set_xticks(np.arange(1, m, s))
    figure.axes[0].set_xlim((1, m-1))
    figure.axes[0].set_yticks(np.linspace(0.0, 0.8, 9).round(2))
    figure.axes[0].set_ylim((0, 0.80))

    labels = ['Наилучшая метрика']
    labels += ['Да', 'Нет']
    labels += ['Значения метрики']
    labels += [f'Процесс {j}: {v:.4f}'
               for j, v
               in (trials
                   .groupby('job')['best']
                   .max()
                   .sort_values()
                   .items())][::-1]

    figure.axes[0].legend(
            handles=figure.axes[0].get_legend_handles_labels()[0],
            labels=labels,
            title='Значения метрики (F1-weighted)',
            loc='lower right',
            alignment='left'
    )

    sns.lineplot(
            data=trials,
            x='index',
            y='best',
            linewidth=1.5,
            style='job',
            hue='target',
            style_order=order,
            hue_order=[True, False],
            ax=figure.axes[1],
            palette=sns.color_palette('hls', 15)[8::5][::-1]
    )

    # Определяем стиль графика 2.
    # Определяем заголовок.
    figure.axes[1].set_title(
            label='Зависимость лучшей метрики F1-weighted от испытания',
            fontsize='large',
            y=1.035
    )

    figure.axes[1].set_xlabel('Номер испытания')
    figure.axes[1].set_ylabel('Максимальное значение метрики F1-weighted')
    # Определяем подписи значений для оси.
    s, m = step(trials['index'].max())
    figure.axes[1].set_xticks(np.arange(1, m, s))
    figure.axes[1].set_xlim((1, m-1))
    figure.axes[1].set_yticks(np.linspace(0.0, 0.8, 9).round(2))
    figure.axes[1].set_ylim((0.0, 0.8))

    figure.axes[1].legend(
            handles=figure.axes[1].get_legend_handles_labels()[0],
            labels=labels,
            title='Значения метрики (F1-weighted)',
            loc='lower right',
            alignment='left'
    )

    for i in range(2):
        for s in 'top', 'right', 'bottom', 'left':
            figure.axes[i].spines[s].set_visible(False)

        # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\studies.png',
            bbox_inches='tight',
            dpi=150
        )

    plt.close(figure)
