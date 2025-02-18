## Llama.cpp b4732 Setup
1. 
```bash {cmd=true}
git clone https://github.com/ggml-org/llama.cpp.git
```
2. 
```bash {cmd=true}
cd llama.cpp
```
3. 
```bash {cmd=true}
git checkout b4732
```
4. 
```bash {cmd=true}
git submodule update --init --recursive
```
5. 
```bash {cmd=true}
cmake .
```
6. 
```bash {cmd=true}
make clean
```
7. 
```bash {cmd=true}
make all -j
```

### Convert .safetensors to .gguf 8bit
```bash
python {convert_hf_to_gguf.py_path} {merged_model_path} --outtype q8_0 --outfile {gguf_model_save_path}
```