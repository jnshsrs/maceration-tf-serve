FROM tensorflow/serving

LABEL version="1.0.1"

RUN apt-get clean

RUN mkdir serving

ADD ./models /models
ADD ./config/model_config /serving/model_config

EXPOSE 8500
EXPOSE 8501

ENTRYPOINT ["tensorflow_model_server", "--model_config_file=/serving/model_config", "--port=8500", "--rest_api_port=8501"]