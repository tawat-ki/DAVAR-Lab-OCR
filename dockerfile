# FROM ubuntu:20.04
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04


# Set the working directory to /app
WORKDIR /app
RUN apt update 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y  python3-opencv libopencv-dev python3-pip git
# Install required dependencies


# Install PyTorch and torchvision
RUN pip install torch torchvision torchaudio


RUN pip install addict cython numpy albumentations==0.3.2 imagecorruptions matplotlib Pillow==6.2.2 six terminaltables pytest pytest-cov pytest-runner mmlvis scipy scikit-learn yapf
RUN pip install Cython==0.29.33
RUN pip install mmpycocotools
RUN pip install nltk lmdb editdistance opencv-python requests onnx SharedArray tqdm pyclipper imgaug==0.3.0 Shapely Polygon3 scikit-image prettytable transformers seqeval Levenshtein networkx bs4 distance apted lxml jsonlines
RUN pip install mmcv-full==1.3.4
RUN pip install mmdet==2.11.0

RUN git clone https://github.com/tawat-ki/DAVAR-Lab-OCR.git
WORKDIR /app/DAVAR-Lab-OCR
RUN git checkout docker

RUN python3 setup.py develop
RUN g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/tp_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/tp_data.cpp -I/usr/include/opencv4
RUN g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/east_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/east_data.cpp -I/usr/include/opencv4
RUN g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.so -fPIC ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.cpp -I/usr/include/opencv4
RUN g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/east_postprocess.so -fPIC ./davarocr/davar_det/core/post_processing/lib/east_postprocess.cpp -I/usr/include/opencv4
RUN g++ -shared -o ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.so -fPIC ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.cpp -I/usr/include/opencv4
RUN g++ -shared -o ./davarocr/davar_table/datasets/pipelines/lib/gpma_data.so -fPIC ./davarocr/davar_table/datasets/pipelines/lib/gpma_data.cpp -I/usr/include/opencv4
RUN apt install -y cmake

# if cuda version -ge 11
WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
RUN sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|' CMakeLists.txt
RUN sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|' CMakeLists.txt
RUN sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|' CMakeLists.txt
WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/models/losses/
RUN sed -i 's|        loss_warpctc = self\.loss_weight \* self\.criterion(log_probs,|        loss_warpctc = self\.loss_weight \* self\.criterion(log_probs\.cpu(),|' warpctc_loss.py
#fi


WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/src
RUN rm ctc_entrypoint.cu
RUN ln -s ctc_entrypoint.cpp ctc_entrypoint.cu
WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
RUN mkdir build;
WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/build
RUN cmake ..
RUN make

WORKDIR /app/DAVAR-Lab-OCR/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/pytorch_binding
RUN python3 setup.py install
WORKDIR /app
RUN apt install neovim -y
