#Python Version
FROM python:3.12-slim

#Set Working Directory inside the container
WORKDIR /app

#Copy Requirements into the set location of the working directory
COPY requirements.txt .

#Install dependencies without storing them in a temp cache file
RUN pip install --no-cache-dir -r requirements.txt

# Copy your notebook/script
COPY . .

# Default command (optional: runs your script)
CMD ["python", "ML_Model.py"]