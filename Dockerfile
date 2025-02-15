FROM python:3.10

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8090

COPY . /app

ENTRYPOINT ["streamlit","run"]

CMD [ "main.py", "--server.port=8090"]

