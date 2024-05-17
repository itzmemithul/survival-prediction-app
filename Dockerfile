# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /survival_model_api

ADD /requirements.txt .
ADD /*.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
ADD /app.py .

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app.py"]
