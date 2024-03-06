import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

from utils.ml.plot.step import step


def metrics(y_test: pd.DataFrame,
            y_predict: pd.DataFrame,
            y_train: pd.DataFrame,
            title: str,
            labels: pd.Series,
            name: str = 'metrics',
            path: str = None) -> None:
    """
    Строит график оценки модели с помощью различных метрик;

    :param y_test: истинные значения классов в тестовой;
    :param y_predict: предсказанные значения классов на тестовой выборке;
    :param y_train: истинные значения классов в тренировочной выборке;
    :param title: заголовок графика;
    :param labels: метки классов;
    :param name: имя файла;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    sns.set_style('white')

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 15)
    )

    # Создаем объект суб-фигуры размером 3 на 1.
    subfigures: list[plt.Figure] = list(figure.subfigures(
        nrows=1,
        ncols=3,
        wspace=0.1,
        width_ratios=[1, 1.5, 4.5]
    ))

    # Создаем 3 сетки для размещения графиков.
    grid_1 = GridSpec(
        nrows=1,
        ncols=1,
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    grid_2 = GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[5.5, 1.0],
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    grid_3 = GridSpec(
        nrows=5,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 0.2],
        wspace=0.3,
        hspace=0.35,
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    # Добавляем оси для графиков в каждую сетку.
    subfigures[0].add_subplot(grid_1[0, 0])
    subfigures[1].add_subplot(grid_2[0, 0])
    subfigures[1].add_subplot(grid_2[0, 1])
    for i in range(5):
        for j in range(4):
            subfigures[2].add_subplot(grid_3[i, j])
    subfigures[2].add_subplot(grid_3[:, 4])

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t=title,
        fontsize='x-large',
        y=1.065,
    )

    # Строим гистограмму баланса классов.
    sns.barplot(
        x=np.sum(y_train, axis=0),
        y=labels,
        ax=subfigures[0].axes[0],
        orient='h',
        color=sns.color_palette('hls', 15)[13]
    )

    # Определяем стиль для суб-фигуры 1.
    # Определяем заголовок.
    subfigures[0].suptitle(
        t='Баланс в обучающей выборке',
        fontsize='large',
        y=1.035
    )
    subfigures[0].axes[0].set_xlabel('Количество объектов ')
    subfigures[0].axes[0].set_ylabel('Жанры видеоигр')
    # Определяем подписи значений для оси.
    subfigures[0].axes[0].set_xticks([])
    subfigures[0].axes[0].set_yticks(
        ticks=np.arange(labels.size),
        labels=labels.replace({
            'Card & Board Game': 'C&B Game',
            'Real Time Strategy': 'RT Strategy',
            'Turn Based Strategy': 'TB Strategy',
            'Point-and-Click': 'P&C '
        }),
        rotation=90,
        verticalalignment='center'
    )

    # Определяем подписи для столбцов гистограммы.
    subfigures[0].axes[0].bar_label(
        container=subfigures[0].axes[0].containers[0],
        fontsize=12,
        padding=5
    )
    # Убираем оси.
    for s in 'top', 'right', 'bottom', 'left':
        subfigures[0].axes[0].spines[s].set_visible(False)

    # Формируем данные для построения тепловой карты с метриками.
    plot_data = pd.DataFrame(
        data=classification_report(
            y_true=y_test,
            y_pred=y_predict,
            target_names=labels,
            zero_division=0.,
            output_dict=True
        )
    ).iloc[:3, :-4].T

    # Сроим тепловую карту с метриками.
    sns.heatmap(
        data=plot_data,
        ax=subfigures[1].axes[0],
        annot=True,
        linewidths=1.0,
        cmap=sns.color_palette("light:#57b9db", as_cmap=True),
        fmt='.2f',
        cbar=True,
        cbar_ax=subfigures[1].axes[1],
        cbar_kws={'ticklocation': 'left'},
        vmin=0,
        vmax=1
    )

    # Определяем стиль для суб-фигуры 2.
    # Определяем заголовок.
    subfigures[1].suptitle(
        t='Метрики классификации каждого класса',
        fontsize='large',
        y=1.035
    )
    # Определяем подписи значений для оси.
    subfigures[1].axes[0].set_yticks([])
    # Определяем подписи для столбцов гистограммы.
    subfigures[1].axes[0].tick_params(
        axis='x',
        labeltop=True,
        labelbottom=True
    )

    # Формируем данные для построения матриц ошибок.
    plot_data = multilabel_confusion_matrix(
        y_true=y_test,
        y_pred=y_predict,
    )

    s, m = step(plot_data.max())

    # Строим матрицы ошибок.
    for i in range(0, 20):
        sns.heatmap(
            data=plot_data[i],
            ax=subfigures[2].axes[i],
            annot=True,
            linewidths=1.0,
            cmap=sns.color_palette("light:#57b9db", as_cmap=True),
            fmt='d',
            cbar=True if i == 19 else False,
            cbar_ax=subfigures[2].axes[i + (1 if i == 19 else 0)],
            cbar_kws={'ticklocation': 'left'},
            vmin=0,
            vmax=m
        )

    # Определяем стиль для суб-фигуры 3.
    # Определяем заголовок.
    subfigures[2].suptitle(
        t='Матрицы ошибок каждого класса',
        fontsize='large',
        y=1.035
    )
    # Определяем заголовки и подписи осей для каждой матрицы ошибок.
    for i in range(20):
        subfigures[2].axes[i].set_title(
            label=labels[i],
            fontsize='large'
        )
        subfigures[2].axes[i].set_xlabel('Predict class')
        subfigures[2].axes[i].set_ylabel('True class')
    # Определяем подписи оси для тепловой шкалы.
    subfigures[2].axes[20].set_yticks(np.arange(0., m + 1, s))

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + fr'\{name}.png',
            bbox_inches='tight',
            dpi=150
        )
