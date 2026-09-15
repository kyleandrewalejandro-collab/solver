FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for torch, cv2, etc
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 libxext6 libxrender-dev \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy solver requirements
COPY requirements-solver.txt .
RUN pip install --no-cache-dir -r requirements-solver.txt

# Copy solver code
COPY cn31_real_solver.py .
COPY Cn31-Solver-main/ ./Cn31-Solver-main/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV SOLVER_PATH=/app/Cn31-Solver-main

# Run solver
CMD ["python", "cn31_real_solver.py"]
