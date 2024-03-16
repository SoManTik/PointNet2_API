
FROM python:3.9

WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip3 install -r requirements.txt
RUN  pip3 install open3d
COPY . .
EXPOSE 3000
CMD ["python3", "main.py"]