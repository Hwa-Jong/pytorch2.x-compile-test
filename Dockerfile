FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
COPY requirements.txt /workspace/requirements.txt

# RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir -r requirements.txt
