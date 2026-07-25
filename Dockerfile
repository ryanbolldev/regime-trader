FROM python:3.11-slim

RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup . .

RUN mkdir -p /app/logs \
    && chown -R appuser:appgroup /app

# appuser has no home dir (--no-create-home); point HOME at /app (already
# writable) so libraries that cache under ~/.cache (yfinance, etc.) don't fail.
ENV HOME=/app

USER appuser

CMD ["python", "main.py"]
