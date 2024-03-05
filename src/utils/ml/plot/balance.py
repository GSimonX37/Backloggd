import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec


def balance(train: pd.DataFrame,
            test: pd.DataFrame,
            labels: pd.Series,
            path: str = None) -> None:
    """
    Строит график баланса классов в тренировочной и тестовой выборках;

    :param train: метки классов в тренировочной выборке;
    :param test: метки классов в тестовой выборке;
    :param labels: названия меток классов;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    sns.set_style('white')

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 10)
    )

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t='Распределение классов в наборе данных',
        fontsize='x-large',
        y=1.05,
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

    plot_data = pd.DataFrame(
        data=pd.concat(objs=[train, test]).values,
        columns=labels
    ).sum().sort_values(ascending=False)

    # Строим график-1.
    sns.barplot(
        x=plot_data.index,
        y=plot_data,
        ax=figure.axes[0],
        color=sns.color_palette('hls', 15)[8]
    )

    # Определяем стиль для графика-1.
    # Определяем заголовок.
    figure.axes[0].set_title(
        label='Распределение классов в наборе данных',
        y=1.05,
        fontsize='large'
    )
    # Определяем подписи для оси.
    figure.axes[0].set_xlabel('Жанры видеоигр')
    figure.axes[0].set_ylabel('Количество классов')
    # Определяем подписи значений для оси.
    figure.axes[0].set_yticks([])
    # Определяем подписи для столбцов гистограммы.
    figure.axes[0].bar_label(
        container=figure.axes[0].containers[0],
        fontsize=12,
        padding=5
    )

    order = plot_data.index
    plot_data = pd.DataFrame(
        data={
            'values': pd.concat(
                objs=[pd.DataFrame(data=train.values,
                                   columns=labels).sum(),
                      pd.DataFrame(data=test.values,
                                   columns=labels).sum()]),
            'sample': (['Тренировочная выборка'] * len(labels) +
                       ['Тестовая выборка'] * len(labels))
        }
    ).reset_index().sort_values(
        by='values',
        ascending=False
    )

    # Строим график-2.
    sns.barplot(
        data=plot_data,
        x='index',
        y='values',
        hue='sample',
        order=order,
        ax=figure.axes[1],
        palette=sns.color_palette('hls', 15)[8::5]
    )

    # Определяем стиль для графика-2.
    # Определяем заголовок.
    figure.axes[1].set_title(
        label='Распределение классов в тренировочной и тестовой выборках',
        y=1.05,
        fontsize='large'
    )
    # Определяем подписи для оси.
    figure.axes[1].set_xlabel('Жанры видеоигр')
    figure.axes[1].set_ylabel('Количество классов')
    # Определяем подписи значений для оси.
    figure.axes[1].set_yticks([])
    # Определяем подписи для столбцов гистограммы.
    figure.axes[1].bar_label(
        container=figure.axes[1].containers[0],
        fontsize=12,
        padding=5
    )
    figure.axes[1].bar_label(
        container=figure.axes[1].containers[1],
        fontsize=12,
        padding=5
    )
    # Определяем легенду
    figure.axes[1].legend(
        title='Разделение выборок',
        loc='upper right',
        alignment='left'
    )

    # Убираем оси для графиков 1 и 2.
    for i in 0, 1:
        figure.axes[i].spines['top'].set_visible(False)
        figure.axes[i].spines['right'].set_visible(False)
        figure.axes[i].spines['bottom'].set_visible(False)
        figure.axes[i].spines['left'].set_visible(False)

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\balance.png',
            bbox_inches='tight',
            dpi=250
        )
