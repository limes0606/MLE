# Stage 1: only used to borrow a ready-made Java 17 runtime (no apt needed).
FROM eclipse-temurin:17-jre AS java

# Stage 2: same Airflow 2.6.1 as Lab 5. The -python3.9 tag gives modern
# pandas / scikit-learn / xgboost wheels; the default 2.6.1 tag is Python 3.7.
FROM apache/airflow:2.6.1-python3.9

USER root

# Java 17 for PySpark
COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$JAVA_HOME/bin
# Avoids the container-hostname resolution crash seen in Assignment 1
ENV SPARK_LOCAL_IP=127.0.0.1
# Lets scripts/ and dags/ import from /opt/airflow/utils
ENV PYTHONPATH=/opt/airflow

WORKDIR /app
COPY requirements.txt ./

USER airflow
RUN pip install --no-cache-dir -r requirements.txt