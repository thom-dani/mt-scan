FROM mt-scan-base:latest
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONPATH=/workspace/build

WORKDIR /workspace