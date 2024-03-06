import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve


def calibration(y_true: pd.DataFrame,
                y_proba: list[pd.DataFrame],
                labels: pd.Series,
                title: str,
                path: str = None) -> None:
    """
    Строит график калибровоности предсказываемых вероятностей модели;

    :param y_true: истинные значения классов;
    :param y_proba: предсказанные вероятности принадлежности к классу "1";
    :param labels: метки классов;
    :param title: заголовок графика;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    sns.set_style('white')

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 18)
    )

    # Создаем объект суб-фигуры размером 5 на 4.
    subfigures: list = list(figure.subfigures(
        nrows=5,
        ncols=4,
        wspace=0.1,
        hspace=0.25
    ))

    # Создаем сетку для размещения графиков.
    grid = GridSpec(
        nrows=2,
        ncols=1,
        hspace=0.5,
        left=0,
        right=1,
        top=1,
        bottom=0,
        height_ratios=[1, 0.75]
    )

    # Добавляем оси для графиков в сетку.
    for i in range(5):
        for j in range(4):
            for k in range(2):
                subfigures[i][j].add_subplot(grid[k, 0])

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t=title,
        fontsize='x-large',
        y=1.05,
    )

    # Строим графики для каждого класса.
    genre = 0
    for i in range(5):
        for j in range(4):
            # Определяем заголовок суб-фигуры.
            subfigures[i][j].suptitle(
                t=labels[genre],
                fontsize='large',
                y=1.1
            )

            # Формируем данные для построения калибровочной кривой.
            proba_true, proba_predict = calibration_curve(
                y_true=y_true.loc[:, genre],
                y_prob=y_proba[genre].iloc[:, 1],
                n_bins=10
            )

            # Строим калибровочную кривую.
            sns.lineplot(
                x=proba_predict,
                y=proba_true,
                ax=subfigures[i][j].axes[0],
                marker='o',
                linewidth=1.5,
                markersize=6,
                color=sns.color_palette('hls', 15)[8],
                label='Калибровка модели'
            )

            # Добавляем линию идеальной калибровки.
            subfigures[i][j].axes[0].axline(
                xy1=(0, 0),
                xy2=(1, 1),
                color='grey',
                linestyle=':',
                label='Наилучшая калибровка'
            )

            # Добавляем подписи к маркерам.
            for marker in range(len(proba_predict)):
                value = round(proba_true[marker], 1)

                subfigures[i][j].axes[0].annotate(
                    text=value,
                    xy=(proba_predict[marker], proba_true[marker]),
                    xytext=(-5, 5),
                    textcoords='offset points',
                    fontsize=10
                )

            # Определяем стиль графика-1.
            axes = subfigures[i][j].axes[0]
            # Определяем подписи для оси.
            axes.set_xlabel('Средняя прогнозируемая вероятность')
            axes.set_ylabel('Доля истинного класса')
            # Определяем подписи значений для осей.
            axes.set_yticks([])
            axes.set_xticks([])
            axes.set_xbound((0, 1))
            # Определяем легенду.
            axes.legend(loc='upper left')

            # Строим гистограмму.
            sns.histplot(
                x=y_proba[genre].iloc[:, 1],
                bins=np.linspace(0.0, 1.0, 11),
                ax=subfigures[i][j].axes[1],
                color=sns.color_palette('hls', 15)[13]
            )
            # Добавляем подписи к гистограмме.
            subfigures[i][j].axes[1].bar_label(
                container=subfigures[i][j].axes[1].containers[0],
                fontsize=10,
                padding=5
            )

            # Определяем стиль графика-2.
            axes = subfigures[i][j].axes[1]
            # Определяем подписи для оси.
            axes.set_xlabel('Прогнозируемая вероятность')
            axes.set_ylabel('Количество')
            # Определяем подписи значений для осей.
            axes.set_xticks(np.linspace(0.0, 1.0, 11).round(2))
            axes.set_xbound((0, 1))
            axes.set_yticks([])

            # Убираем оси графиков 1 и 2.
            for axes in 0, 1:
                for s in 'top', 'right', 'bottom', 'left':
                    subfigures[i][j].axes[axes].spines[s].set_visible(False)

            genre += 1

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\calibration.png',
            bbox_inches='tight',
            dpi=150
        )
