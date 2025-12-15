FROM ghcr.io/astral-sh/uv:bookworm
LABEL authors="daniel"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      texlive-latex-recommended \
      texlive-latex-extra \
      texlive-fonts-recommended \
      dvipng \
      ghostscript \
      fonts-linuxlibertine \
      cm-super \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# COPY ./src /app/src
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

# Default entrypoint runs the project's main CLI. Use docker-compose or `docker run --entrypoint ...` to override.
ENTRYPOINT ["uv", "run", "./src/main.py"]

# Prints help when run with no args
CMD ["--help"]
