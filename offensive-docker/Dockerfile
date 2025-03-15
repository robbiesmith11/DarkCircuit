FROM aaaguirrep/offensive-docker

# Install OpenSSH server if not already installed
RUN apt-get update && apt-get install -y openssh-server

# Set up SSH
RUN mkdir -p /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise, user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Save and load command history in your local environment
RUN sed -i '1i export HISTFILE="/history/.zsh_history"' /root/.zshrc

# Start SSH service when container launches
CMD service ssh start && /bin/zsh