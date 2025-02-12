## Ubuntu 22.04 x64
1. wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
2. bash Anaconda3-2024.10-1-Linux-x86_64.sh
3. source ~/.bashrc
4. conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
5. conda activate unsloth_env
6. pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
7. pip install --no-deps trl peft accelerate bitsandbytes

Remove or comment
```
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
```
in /root/anaconda3/envs/unsloth_env/lib/python3.11/site-packages/unsloth/__init__.py:49
