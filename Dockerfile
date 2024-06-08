FROM nvidia/cuda:12.1.0-devel-ubuntu18.04
  
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y wget && \
    wget  https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda3-2023.09-0-Linux-x86_64.sh

SHELL ["/bin/bash", "-c"]

RUN source /opt/conda/etc/profile.d/conda.sh && conda create -n hippo python=3.9 -y && \
    conda init bash && echo "conda activate hippo" >> ~/.bashrc


RUN source /opt/conda/etc/profile.d/conda.sh && conda activate hippo && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    pip install pandarallel pandas jupyter numpy datasets sentencepiece openai wandb tiktoken transformers accelerate sentencepiece 

RUN pip install peft bitsandbytes==0.40.2 trl==0.4.7 spicy gradio spaces

WORKDIR /workspace


ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate hippo && /bin/bash"]
