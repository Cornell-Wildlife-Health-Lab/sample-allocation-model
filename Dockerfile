FROM python:3.9-slim

# Install build tools, plus wget & unzip for IPOPT
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      unzip \
      coinor-libipopt-dev \
 && rm -rf /var/lib/apt/lists/*

#  Download & unzip IPOPT (Linux64)
RUN wget -qO /tmp/ipopt.zip \
      "https://matematica.unipv.it/gualandi/solvers/ipopt-linux64.zip" \
 && unzip -q /tmp/ipopt.zip -d /opt \
 && rm /tmp/ipopt.zip


# Expose the IPOPT executable on PATH
RUN chmod +x /opt/ipopt \
 && ln -sf /opt/ipopt /usr/local/bin/ipopt

# 4) sanity check
RUN ipopt --version

# Make a folder to hold model files
WORKDIR /app

# Copy your model files
COPY src /app

# Install python packages as defined in a requirements file
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
       -r /app/software/requirements.txt


CMD ["/bin/sh", "./scripts/run_scripts.sh"]