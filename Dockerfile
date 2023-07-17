FROM python:3.8

WORKDIR /app

COPY . .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]