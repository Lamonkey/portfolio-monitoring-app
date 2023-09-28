FROM python:3.11

# Directory to store the code
RUN mkdir /code

# Set up a directory for the persistent volume
VOLUME code/instance

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# make database writable
RUN chmod 777 /code/instance


# Start the Panel app
# EXPOSE 7860

WORKDIR /code/src
# CMD ["panel", "serve", "editingPage.py", "indexPage.py","--basic-auth", "../instance/credential.json", "--setup","backgroundTask.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin", "*"]
# for huggingface
CMD ["panel", "serve", "editingPage.py", "indexPage.py","--setup","backgroundTask.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin", "jc-software-risk-monitor-app.hf.space"]

# permission required by huggingface
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir .chroma
RUN chmod 777 .chroma