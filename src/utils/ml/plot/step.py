def step(maximum: int | float, k: int | float = 1) -> (int, int):
    """
    Вычисляет максимальное значение и шаг меток для оси;

    :param maximum: максимальное значение измеряемой величины;
    :param k: коэффициент масштаба;
    :return: шаг меток и максимальное значение для оси.
    """

    if maximum > 500_000:
        s = int(50_000 * k)
    elif maximum > 100_000:
        s = int(10_000 * k)
    elif maximum > 50_000:
        s = int(5_000 * k)
    elif maximum > 20_000:
        s = int(2_000 * k)
    elif maximum > 10_000:
        s = int(1_000 * k)
    elif maximum > 5_000:
        s = int(500 * k)
    elif maximum > 2_000:
        s = int(200 * k)
    elif maximum > 1_000:
        s = int(100 * k)
    elif maximum > 500:
        s = int(50 * k)
    elif maximum > 200:
        s = int(20 * k)
    elif maximum > 100:
        s = int(10 * k)
    elif maximum > 50:
        s = int(5 * k)
    elif maximum > 20:
        s = int(2 * k)
    else:
        s = int(1 * k)

    maximum = ((maximum // s) + 1) * s

    return s, maximum
