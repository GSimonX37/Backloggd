import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec


def scalability(train_sizes: pd.Series,
                train_scores: pd.DataFrame,
                test_scores: pd.DataFrame,
                fit_times: pd.DataFrame,
                score_times: pd.DataFrame,
                title: str,
                path: str = None) -> None:
    """
    Строит график масштабируемости модели;

    :param train_sizes: размеры тренировочных данных;
    :param train_scores: оценка метрики модели не тренировочной выборке;
    :param test_scores: оценка метрики модели не тестовой выборке;
    :param fit_times: время обучения модели;
    :param score_times: время предсказания модели;
    :param title: заголовок графика;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    sns.set_style('white')

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 15)
    )

    # Создаем сетку для размещения графиков.
    gird = GridSpec(
        figure=figure,
        nrows=3,
        ncols=2,
        hspace=0.05
    )

    # Добавляем оси для графиков в сетку.
    figure.add_subplot(gird[0, :])
    figure.add_subplot(gird[1, 0])
    figure.add_subplot(gird[1, 1])
    figure.add_subplot(gird[2, 0])
    figure.add_subplot(gird[2, 1])

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t=title,
        fontsize='x-large',
        y=1.025
    )

    # Формируем данные для построения графиков.
    size = train_sizes.size
    scores = [*train_scores.mean(axis=1)] + [*test_scores.mean(axis=1)]
    all_fit_times = [*fit_times.mean(axis=1)] + [*fit_times.mean(axis=1)]
    all_score_times = [*train_scores.mean(axis=1)] + [*score_times.mean(axis=1)]
    types = ['Тренировочные данные'] * size + ['Тестовые данные'] * size
    plot_data = pd.DataFrame(
        data={
            'train_sizes': [*train_sizes] * 2,
            'scores': scores,
            'fit_times': all_fit_times,
            'score_times': all_score_times,
            'type': types
        }
    )

    # Строим линейную диаграмму train_sizes - scores.
    sns.lineplot(
        data=plot_data,
        x='train_sizes',
        y='scores',
        hue='type',
        marker='o',
        linewidth=2,
        markersize=8,
        ax=figure.axes[0],
        palette=sns.color_palette('hls', 15)[8::5]
    )

    # Определяем стиль для графика 1.
    # Определяем заголовок.
    figure.axes[0].set_title(
        label='Кривые обучения',
        fontsize='large',
        y=1
    )
    # Определяем подписи для осей.
    figure.axes[0].set_xlabel('Размер выборки (%)')
    figure.axes[0].set_ylabel('Метрика F1-weighted')
    # Определяем подписи значений для оси.
    figure.axes[0].set_yticks(np.linspace(0.0, 1.0, 11).round(1))
    figure.axes[0].set_ylim((-0.05, 1.05))
    figure.axes[0].set_xticks(np.arange(0, 101, 5))
    figure.axes[0].set_xlim((-1, 101))
    # Определяем подписи значений маркеров.
    for t in 'Тренировочные данные', 'Тестовые данные':
        x_array = plot_data[plot_data['type'] == t].loc[:, 'train_sizes']
        y_array = plot_data[plot_data['type'] == t].loc[:, 'scores']
        for x, y in zip(x_array, y_array):
            value = round(y, 2)
            figure.axes[0].annotate(
                text=value,
                xy=(x, y),
                xytext=(-10, 10 if t == 'Тестовые данные' else -15),
                textcoords='offset points',
                fontsize=12
            )
    # Определяем стандартные отклонения.
    for scores, color in zip((train_scores, test_scores), (8, 13)):
        figure.axes[0].fill_between(
            x=train_sizes,
            y1=scores.mean(axis=1) - scores.std(axis=1),
            y2=scores.mean(axis=1) + scores.std(axis=1),
            color=sns.color_palette('hls', 15)[color],
            alpha=0.1,
        )
    # Определяем легенду.
    figure.axes[0].legend(
        title='Метрика F1-weighted',
        loc='lower right',
        alignment='left'
    )

    # Строим линейную диаграмму train_sizes - fit_times.
    sns.lineplot(
        x=train_sizes,
        y=fit_times.mean(axis=1),
        ax=figure.axes[1],
        marker='o',
        linewidth=2,
        markersize=8,
        color=sns.color_palette('hls', 15)[8]
    )

    # Определяем стиль для графика 2.
    # Определяем заголовок.
    figure.axes[1].set_title(
        label='Время обучения модели',
        fontsize='large',
        y=1
    )
    # Определяем подписи для осей.
    figure.axes[1].set_xlabel('Размер выборки (%)')
    figure.axes[1].set_ylabel('Время обучения (сек.)')
    # Определяем подписи значений для оси.
    figure.axes[1].set_xticks(np.arange(0, 101, 10))
    figure.axes[1].set_xlim((-1, 101))
    # Определяем подписи значений маркеров.
    for x, y in zip(train_sizes, fit_times.mean(axis=1)):
        value = round(y, 2)
        figure.axes[1].annotate(
            text=value,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords='offset points',
            fontsize=12
        )

    # Строим линейную диаграмму train_sizes - score_times.
    sns.lineplot(
        x=train_sizes,
        y=score_times.mean(axis=1),
        ax=figure.axes[2],
        marker='o',
        linewidth=2,
        markersize=8,
        color=sns.color_palette('hls', 15)[13]
    )

    # Определяем стиль для графика 3.
    # Определяем заголовок.
    figure.axes[2].set_title(
        label='Время предсказания модели',
        fontsize='large',
        y=1
    )
    # Определяем подписи для осей.
    figure.axes[2].set_xlabel('Размер выборки (%)')
    figure.axes[2].set_ylabel('Время предсказания (сек.)')
    # Определяем подписи значений для оси.
    figure.axes[2].set_xticks(np.arange(0, 101, 10))
    figure.axes[2].set_xlim((-1, 101))
    # Определяем подписи значений маркеров.
    for x, y in zip(train_sizes, score_times.mean(axis=1)):
        value = round(y, 2)
        figure.axes[2].annotate(
            text=value,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords='offset points',
            fontsize=12
        )
    # Определяем стандартные отклонения для графиков 2 и 3.
    for i, (times, color) in enumerate(zip((fit_times, score_times), (8, 13))):
        figure.axes[i + 1].fill_between(
            x=train_sizes,
            y1=times.mean(axis=1) - times.std(axis=1),
            y2=times.mean(axis=1) + times.std(axis=1),
            color=sns.color_palette('hls', 15)[color],
            alpha=0.1,
        )

    # Строим точечную диаграмму fit_times - test_scores.
    sns.scatterplot(
        data=plot_data.loc[plot_data['type'] == 'Тестовые данные', :],
        x='fit_times',
        y='scores',
        size='train_sizes',
        sizes=(20, 200),
        legend='full',
        ax=figure.axes[3],
        color=sns.color_palette('hls', 15)[8],
    )

    # Определяем стиль для графика 4.
    # Определяем заголовок.
    figure.axes[3].set_title(
        label='Зависимость качества модели от времени обучения',
        fontsize='large',
        y=1
    )
    # Определяем подписи для осей.
    figure.axes[3].set_xlabel('Время обучения (сек.)')
    figure.axes[3].set_ylabel('Метрика F1-weighted')
    # Определяем подписи значений маркеров.
    for x, y in zip(fit_times.mean(axis=1), test_scores.mean(axis=1)):
        value = round(y, 2)
        figure.axes[3].annotate(
            text=value,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords='offset points',
            fontsize=12
        )
    # Определяем легенду.
    figure.axes[3].legend(
        handles=figure.axes[3].get_legend_handles_labels()[0],
        labels=[f'{int(size)}%' for size in train_sizes],
        title='Размер выборки',
        loc='lower right',
        alignment='left'
    )

    # Строим точечную диаграмму score_times - test_scores.
    sns.scatterplot(
        data=plot_data.loc[plot_data['type'] == 'Тестовые данные', :],
        x='score_times',
        y='scores',
        size='train_sizes',
        sizes=(20, 200),
        legend='full',
        ax=figure.axes[4],
        color=sns.color_palette('hls', 15)[13],
    )

    # Определяем стиль для графика 5.
    # Определяем заголовок.
    figure.axes[4].set_title(
        label='Зависимость качества модели от времени предсказания',
        fontsize='large',
        y=1
    )
    # Определяем подписи для осей.
    figure.axes[4].set_xlabel('Время предсказания (сек.)')
    figure.axes[4].set_ylabel('Метрика F1-weighted')
    # Определяем подписи значений маркеров.
    for x, y in zip(score_times.mean(axis=1), test_scores.mean(axis=1)):
        value = round(y, 2)
        figure.axes[4].annotate(
            text=value,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords='offset points',
            fontsize=12
        )
    # Определяем легенду.
    figure.axes[4].legend(
        handles=figure.axes[4].get_legend_handles_labels()[0],
        labels=[f'{int(size)}%' for size in train_sizes],
        title='Размер выборки',
        loc='lower right',
        alignment='left'
    )

    # Убираем оси графиков 1 - 5.
    for i in range(5):
        for s in 'top', 'right', 'bottom', 'left':
            figure.axes[i].spines[s].set_visible(False)

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\scalability.png',
            bbox_inches='tight',
            dpi=500
        )
