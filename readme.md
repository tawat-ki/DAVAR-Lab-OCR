# DAVAR-OCR
### installation 
Please note that the installation should be performed while the Docker container is running, and it is not possible to install it using the Dockerfile.

```bash
git clone https://github.com/tawat-ki/DAVAR-Lab-OCR.git
cd DAVAR-Lab-OCR
git checkout docker
docker build -t davarocr .
docker run -it --gpus 1  -v .:/app/DAVAR-Lab-OCR davarocr bash
bash setup.py
```

