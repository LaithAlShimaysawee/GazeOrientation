FROM python:3.10

ADD main.py /

RUN pip install numpy
RUN pip install opencv-python
RUN pip install mediapipe

CMD [ "python", "./main.py" ]