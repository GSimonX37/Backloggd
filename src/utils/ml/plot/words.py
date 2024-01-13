import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import CountVectorizer

from utils.ml.plot.step import step


def words(data: pd.DataFrame,
          labels: pd.Series,
          stop_words,
          path: str | None = None) -> None:
    """
    Строит график частоты слов;

    :param data: данные с текстом;
    :param labels: метки классов;
    :param stop_words: стоп-слова;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    sns.set_style('white')

    counter = CountVectorizer(
        stop_words=stop_words
    )

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 20)
    )

    # Создаем объект суб-фигуры размером 3 на 1.
    subfigures: list[plt.Figure] = list(figure.subfigures(
        nrows=3,
        ncols=1,
        hspace=0.1,
        height_ratios=[0.5, 1.5, 1]
    ))

    # Создаем сетки для размещения графиков.
    grid_1 = GridSpec(
        figure=subfigures[1],
        nrows=5,
        ncols=4,
        hspace=0.05,
        wspace=0.05,
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    gird_2 = GridSpec(
        figure=subfigures[2],
        nrows=1,
        ncols=2,
        wspace=0.01,
        width_ratios=[1, 0.025]
    )

    # Добавляем оси для графиков.
    subfigures[0].add_subplot()
    for i in range(5):
        for j in range(4):
            subfigures[1].add_subplot(grid_1[i, j])
    subfigures[2].add_subplot(gird_2[0, 0])
    subfigures[2].add_subplot(gird_2[0, 1])

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t='Частота слов в описаниях видеоигр',
        fontsize='x-large',
        y=1.025,
    )

    number = (counter
              .fit_transform(data['description'])
              .sum(axis=0))

    plot_data = pd.Series(
        data=number.tolist()[0],
        index=counter.get_feature_names_out()
    ).sort_values(ascending=False)[:20]

    # Строим гистограмму частоты слов для всех жанров.
    sns.barplot(
        x=plot_data.index,
        y=plot_data,
        ax=subfigures[0].axes[0],
        color=sns.color_palette('hls', 15)[8]
    )

    # Определяем стиль для суб-фигуры 1.
    # Определяем заголовок.
    subfigures[0].suptitle(
        t='Частота слов в описаниях видеоигр для всех жанров',
        fontsize='large',
        y=1.05
    )
    # Определяем подписи для оси.
    subfigures[0].axes[0].set_xlabel('Слова')
    subfigures[0].axes[0].set_ylabel('Количество')
    # Определяем подписи значений для оси.
    subfigures[0].axes[0].set_yticks([])
    # Определяем подписи для столбцов гистограммы.
    subfigures[0].axes[0].bar_label(
        container=subfigures[0].axes[0].containers[0],
        fontsize=12,
        padding=5
    )
    # Убираем оси.
    for s in 'top', 'right', 'bottom', 'left':
        subfigures[0].axes[0].spines[s].set_visible(False)

    # Получаем данные для построения графика.
    plot_data = pd.DataFrame()
    for i in range(20):
        number = (counter
                  .fit_transform(data.loc[data[i] == 1, 'description'])
                  .sum(axis=0))

        number_words = pd.Series(
            data=number.tolist()[0],
            index=counter.get_feature_names_out()
        ).sort_values(ascending=False)[:5]

        plot_data = plot_data.join(
            other=pd.DataFrame(
                data=number_words,
                index=number_words.index,
                columns=[labels[i]]
            ),
            how='outer'
        )

    # Строим гистограмму частоты слов для каждого жанра.
    for i in range(20):
        number = (counter
                  .fit_transform(data.loc[data[i] == 1, 'description'])
                  .sum(axis=0))

        total = pd.Series(
                data=number.tolist()[0],
                index=counter.get_feature_names_out()
        ).sort_values(ascending=False)[:5]

        sns.barplot(
            x=total.index,
            y=total,
            ax=subfigures[1].axes[i],
            color=sns.color_palette('hls', 15)[13]
        )

    # Определяем стиль для суб-фигуры 3.
    # Определяем заголовок.
    subfigures[1].suptitle(
        t='Частота слов в описаниях видеоигр для каждого жанра',
        fontsize='large',
        y=1.05
    )
    for i in range(20):
        subfigures[1].axes[i].set_title(
            label=labels[i],
            fontsize='large',
            y=1.05
        )
        # Определяем подписи значений для оси.
        subfigures[1].axes[i].set_xlabel('Слова')
        subfigures[1].axes[i].set_ylabel('Количество')
        # Определяем подписи значений для оси.
        subfigures[1].axes[i].set_yticks([])
        # Определяем подписи для столбцов гистограммы.
        subfigures[1].axes[i].bar_label(
            container=subfigures[1].axes[i].containers[0],
            fontsize=10,
            padding=5
        )
        # Убираем оси.
        for s in 'top', 'right', 'bottom', 'left':
            subfigures[1].axes[i].spines[s].set_visible(False)

    indexes = (plot_data.agg(['count', 'sum'], axis=1)
               .sort_values(by=['count', 'sum'], ascending=False)
               .index)
    plot_data = plot_data.loc[indexes, :].iloc[:10, :]
    plot_data = plot_data.rename(
        columns={
            'Card & Board Game': 'C&B Game',
            'Real Time Strategy': 'RT Strategy',
            'Turn Based Strategy': 'TB Strategy'
        }
    )

    sns.heatmap(
        data=plot_data,
        ax=subfigures[2].axes[0],
        annot=True,
        linewidths=1.0,
        cmap=sns.color_palette("light:#57b9db", as_cmap=True),
        fmt='.0f',
        cbar=True,
        cbar_ax=subfigures[2].axes[1],
        cbar_kws={'ticklocation': 'left'},
        vmin=0,
        vmax=step(plot_data.max(axis=None), 2)[-1]
    )

    # Определяем стиль для суб-фигуры 2.
    # Определяем заголовок.
    subfigures[2].suptitle(
        t=('Частота слов, присутствующих в описаниях видеоигр '
           'в нескольких жанрах'),
        fontsize='large',
        y=1.05
    )
    # Определяем подписи значений для оси.
    subfigures[2].axes[0].set_xlabel('Жанры видеоигр')
    subfigures[2].axes[0].set_ylabel('Слова')
    # Определяем подписи значений для оси.
    s, m = step(plot_data.max(axis=None), 2)
    subfigures[2].axes[1].set_ybound(0, m)
    subfigures[2].axes[1].set_yticks(np.arange(0, m + 1, s))
    subfigures[2].axes[0].tick_params(
        axis='x',
        labelbottom=True,
        rotation=0
    )
    subfigures[2].axes[0].tick_params(
        axis='y',
        labelbottom=True,
        rotation=90
    )

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\words.png',
            bbox_inches='tight',
            dpi=500
        )
