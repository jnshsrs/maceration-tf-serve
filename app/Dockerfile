FROM tensorflow/tensorflow:2.9.1
COPY . .

RUN apt update
RUN apt-get update -y
RUN apt-get install libgl1 -y

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD [ "python", "./app.py"]
