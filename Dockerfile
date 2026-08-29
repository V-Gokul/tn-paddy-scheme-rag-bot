# 1. Start from a small official Python image
FROM python:3.11-slim

# 2. All following commands run inside this folder
WORKDIR /app

# 3. Copy ONLY requirements first - this makes rebuilds much faster
COPY requirements.txt .

# 4. Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Now copy the rest of the source code
COPY . .
RUN chmod +x entrypoint.sh

# 6. Document which port the app listens on (Streamlit's default)
EXPOSE 8501

# 7. Basic container health check against Streamlit's health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 8. Build the vector DB on first run if missing, then start the app
ENTRYPOINT ["./entrypoint.sh"]
