# Use the official NVIDIA base image for PyTorch with CUDA support
FROM nvcr.io/nvidia/pytorch:21.06-py3

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
