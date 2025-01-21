FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install keras_image_helper
RUN pip install tflite-runtime
RUN pip install flask
RUN pip install gunicorn
RUN pip install numpy==1.26.4

COPY ../hair-classifier.tflite .
COPY predict.py .

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]