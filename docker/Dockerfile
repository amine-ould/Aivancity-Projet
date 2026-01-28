FROM mambaorg/micromamba:1.5.3

WORKDIR /app

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n ml-predictive-maintenance -f /tmp/environment.yml \
    && micromamba clean -a -y

ENV PATH=/opt/conda/envs/ml-predictive-maintenance/bin:$PATH
ENV PYTHONUNBUFFERED=1

COPY . /app

CMD ["python", "run_training.py"]
