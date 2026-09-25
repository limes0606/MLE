FROM jupyter/pyspark-notebook:spark-3.5.0

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV JUPYTER_ENABLE_LAB=yes

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]