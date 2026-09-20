FROM flask-tex-base:latest
WORKDIR /app

# LilyPond (provides lilypond-book) is only needed by this app, so it goes here,
# layered on top of the shared TeX base rather than in the base itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
    lilypond \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
