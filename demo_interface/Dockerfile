
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    google-cloud-storage \
    google-genai

COPY demo_interface/ /app/

ENV PORT=8080

CMD ["streamlit", "run", "dashboard_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8080", \
     "--server.headless=true"]
