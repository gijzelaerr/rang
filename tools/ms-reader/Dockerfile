# Build with --platform linux/amd64: casacore's public wheel is x86-64.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Optional, isolated reader. No SKA or private package registry is used.
RUN python -m pip --isolated install --no-cache-dir --only-binary=:all: \
    --index-url https://pypi.org/simple numpy==2.5.3 python-casacore==3.8.1

WORKDIR /work
ENTRYPOINT ["python"]
