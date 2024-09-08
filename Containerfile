FROM texlive/texlive:latest
WORKDIR /project
COPY scripts/requirements.txt scripts/requirements.txt
RUN apt-get update && apt-get install -y \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install -r scripts/requirements.txt --no-cache --break-system-packages
RUN python -c "import matplotlib.font_manager; matplotlib.font_manager.findfont('dummy', rebuild_if_missing=True)"
