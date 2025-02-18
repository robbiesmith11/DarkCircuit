# Start from the official Ollama image
FROM ollama/ollama

# Set working directory
WORKDIR /ollama

# Install curl
RUN apt update && apt install -y curl

# Copy the model list file and entrypoint script
COPY models.txt /ollama/models.txt
COPY entrypoint.sh /ollama/entrypoint.sh

# Ensure the script is executable
RUN chmod +x /ollama/entrypoint.sh

# Expose Ollama API port
EXPOSE 11434

# Set entrypoint script
ENTRYPOINT ["/bin/bash", "/ollama/entrypoint.sh"]
