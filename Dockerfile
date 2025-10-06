# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container (/app)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all
COPY . .

# so serve as API
EXPOSE 5000 

CMD ["python", "train.py"]