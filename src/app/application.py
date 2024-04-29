import pandas as pd

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response

from app.model import Model
from utils.data import preprocess


model = Model()

app = FastAPI()


@app.post("/")
async def send_response(request: Request) -> Response:
    """
    Обрабатывает запрос клиента, который должен содержать:

    - description: описание видеоигры на английском языке;
    - threshold: порог принятия решения о принадлежности объекта к классу.

    Отправляет ответ на запрос клиента, который содержит:

    - description: описание видеоигры на английском языке;
    - threshold: порог принятия решения о принадлежности объекта
      к положительному классу;
    - genres: вероятности, с которыми видеоигра принадлежит к игровым жанрам;

    :param request: запрос от клиента;
    :return: ответ на запрос клиента.
    """

    data: dict = await request.json()

    body = {}

    description = data['description']
    threshold = data['threshold']

    description = pd.Series(
        data=[description],
        index=[0],
        name='description'
    )

    description = preprocess(description)

    genres = model.result(
        description=description,
        threshold=threshold
    )

    body['description'] = data['description']
    body['threshold'] = data['threshold']
    body['genres'] = genres

    return Response(
        status_code=200,
        media_type='application/json',
        content=f'{body}'
    )
