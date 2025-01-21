FROM svizor/zoomcamp-model:3.11.5-slim


RUN pip install keras_image_helper
RUN pip install tflite-runtime
RUN pip install gradio
RUN pip install numpy==1.26.4

COPY hair-classifier.tflite .
COPY deploy_gradio.py .

EXPOSE 7860

CMD ["python", "deploy_gradio.py"]