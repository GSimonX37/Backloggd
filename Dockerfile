FROM python:3.10

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./models/complement ./models/complement
COPY ./src/app ./app
COPY ./src/run.py .
COPY ./src/utils/data ./utils/data

CMD ["python", "run.py", "complement"]