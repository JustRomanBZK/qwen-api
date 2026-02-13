FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

ENV HOST=0.0.0.0
ENV PORT=8001
EXPOSE 8001

HEALTHCHECK --interval=60s --timeout=15s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8001/health || exit 1

ENTRYPOINT ["python3", "main.py"]
