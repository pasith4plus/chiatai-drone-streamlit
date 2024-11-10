# docker build -t pasithbas4plus/llm-demo:tagname .
# docker login
# docker push pasithbas4plus/llm-demo:tagname

FROM python:3.10-slim 

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
# WORKDIR APP_HOME /app
COPY . ./

ENV PORT 8081

RUN pip install --no-cache-dir -r requirements.txt

# As an example here we're running the web service with one worker on uvicorn.
# CMD exec uvicorn app.app:app --host 0.0.0.0 --port ${PORT} --workers 1