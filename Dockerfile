FROM python:3.7

EXPOSE 8080

COPY requirements.txt stormlight/requirements.txt

RUN pip3 install -r stormlight/requirements.txt

COPY . stormlight

RUN pip3 install -r stormlight/requirements.txt && pip3 install -e stormlight/lib

CMD cd stormlight && streamlit run --server.port 8080 --server.enableCORS false services/app.py data/models/v5

