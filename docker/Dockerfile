FROM mt-scan-base:latest

ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID appuser && \
    useradd -u $UID -g $GID -m appuser

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

USER appuser
ENV HOME=/home/appuser
ENV PYTHONPATH=/workspace/build
WORKDIR /workspace