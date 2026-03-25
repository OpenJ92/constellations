FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

CMD ["bash"]
