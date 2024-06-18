FROM python:3.11.6-slim-bookworm as base

WORKDIR /code

RUN groupadd --gid 1000 appgroup
RUN adduser -u 1000 --gid 1000 api
RUN chown -R api:appgroup /code

USER api

COPY requirements.txt /tmp/
RUN pip install --upgrade pip 
RUN pip install --default-timeout=5000  -r /tmp/requirements.txt
RUN pip install uvicorn==0.29.0 loguru==0.7.2

COPY --chown=api:appgroup --chmod=700 ./main.py /code/main.py
COPY --chown=api:appgroup --chmod=700 ./src /code/src

ARG MODEL_NAME
ENV MODEL_NAME=${MODEL_NAME}

ENV PATH=$PATH:/home/api/.local/bin

ENTRYPOINT uvicorn main:app \
            --host 0.0.0.0  \
            --port 8084 \
            --log-level info \
            --workers=1