FROM python:3.7
COPY . /app
RUN pip install -U spacy
RUN pip install -U nltk
RUN python3 -m spacy download en

CMD python3 /app/run_preparing_data.py