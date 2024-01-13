def remove(genres: list):
    """
    Удаляет некоторые жанры из списка;

    :param genres: список жанров;
    :return: список жанров.
    """

    if 'MOBA' in genres:
        genres.remove('MOBA')
    if 'Pinball' in genres:
        genres.remove('Pinball')

    return genres
