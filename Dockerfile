FROM python:3.10

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./models/sgd ./models/sgd
COPY ./src/app ./app
COPY ./src/run.py .
COPY ./src/utils/ml/preprocessing.py ./utils/ml/preprocessing.py

CMD ["python", "run.py", "sgd"]