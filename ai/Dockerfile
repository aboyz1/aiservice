# Use the official Python 3.12 image
FROM python:3.12.0-slim

# Set the working directory inside the container
WORKDIR /usr/src/ai

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies listed in the requirements.txt file
RUN pip install --default-timeout=1000000 --no-cache-dir -r requirements.txt

# Copy the application source code to the container
COPY ./app ./app

# Create a mount point for generated documents
VOLUME /usr/src/ai/generated_docs

# Expose port 8000 for the Flask app
EXPOSE 8000

# Add after EXPOSE line in Dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ai || exit 1

# Run Flask app directly, no need to use `flask run` since we call `app.run()` in `main.py`
CMD ["python", "app/main.py"]