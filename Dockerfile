FROM python:3.10-slim

#set working directory for container
WORKDIR /app

COPY requirements.txt .

# --no-cache-dir → reduces image size (no pip cache stored)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Defines the default command when container starts
# Breakdown:
# uvicorn → server
# main:app
# main = filename (main.py)
# app = FastAPI instance inside it
# --host 0.0.0.0 → allows external access
# --port 8000 → runs on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]