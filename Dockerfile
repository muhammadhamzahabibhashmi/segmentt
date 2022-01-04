FROM nvidia/cuda:10.1-base-ubuntu18.04

WORKDIR /seggg

RUN apt-get -qq update && apt-get -y install \
    cmake \
    python3.7 \ 
    ghostscript \
    git \
    libffi-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-turbo-progs \
    libjpeg8-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libssl-dev \
    libsqlite3-dev \
    libtiff5-dev \
    libwebp-dev \
    netpbm \
    sudo \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*




RUN apt-get install libtiff5-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    python3-pip \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

COPY . /seggg

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# RUN python3 segmenttt/setup.py install
RUN pip3 install --upgrade setuptools
RUN pip3 install Cython
RUN pip3 install scikit-build
# RUN pip3 install opencv-contrib-python opencv-python
RUN pip3 install pyarrow==0.15.1
RUN pip3 install streamlit
RUN pip3 install -r requirements.txt
RUN pip3 install -U torch==1.6 torchvision==0.7 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

# RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mit-semseg==1.0.0
RUN pip3 install timm==0.4.12
RUN pip3 install einops 
RUN pip3 install openmim
# RUN mim install mmcv-full
RUN pip3 install mmcv-full
RUN pip3 install git+git://github.com/open-mmlab/mmsegmentation.git
#RUN python3 setup.py install 


RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
#docker image build -t st:app .
EXPOSE 8501

ENTRYPOINT ["streamlit", "run" , "final_code.py"] 
