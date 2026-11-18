From python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY fast_api.py .
COPY streamlit_app.py .
COPY style.css .

EXPOSE 8000 
EXPOSE 8501

CMD ["sh", "-c", "uvicorn fast_api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]

