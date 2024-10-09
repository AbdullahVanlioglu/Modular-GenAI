# Modular-GenAI
> :warning: Under Development

Modular-GenAI is a library designed for Generative AI (GenAI) research. Our goal is to provide a flexible, clean and easy-to-use set of components that can be combined in various ways to enable experimentation with different generative AI algorithms. The repository includes both PyTorch and Jax implementations of the algorithms.


### **Transformers**

| Transformers |  <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 64px alt="logo"></img> | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width = 50px  height = 50px alt="logo"></img> |
|:-----:|:---------:|:---------:|
|**LLM Transformers**| | |
|  Mistral  |:x:|:x:|
|  Llama v3.2  |:x:|:x:|
|  Llama v2  |:x:|:heavy_check_mark:|
|**Decision Transformers**| | |
|  Behavior Clonning  |:x:|:heavy_check_mark:|
|  Decision Transformer  |:x:|:heavy_check_mark:|
|  Q-Transformer  |:x:|:x:|
|**Vision Transformers**| | |
|  ViT  |:x:|:x:|
|  MaxViT  |:x:|:x:|
|**Diffusion Transformers**| | |
|  DiT  |:x:|:x:|
|**Multimodal Transformers**| | |
|  Qwen2.5-VL  |:x:|:x:|
|**Other Specialized Transformers**| | |
|  Mamba  |:x:|:x:|


### **Diffusion Models**

| Diffusion Models | <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 64px alt="logo"></img> | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width = 50px  height = 50px alt="logo"></img> |
|:-----:|:---------:|:---------:|
|**Image Generation**| | |
|  Stable Diffusion  |:x:|:heavy_check_mark:|
|**Audio Generation (Optional)**| | |
|  AudioLDM  |:x:|:x:|
|**Video Generation**| | |
|  Video Diffusion Models  |:x:|:x:|
|**3D Generation (Optional)**| | |
|  DreamFusion  |:x:|:x:|


### **Components**

| Components | <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" width = 64px alt="logo"></img> | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width = 50px  height = 50px alt="logo"></img> |
|:-----:|:---------:|:---------:|
|**Fine Tuning**| | |
| PPO |:x:|:x:|
| DPO |:x:|:x:|
| LoRA |:x:|:x:|
| Reinforce Style Optimization |:x:|:x:|
| Reverse Curriculum Reinforcement Learning |:x:|:x:|
|**Attention Modules**| | |
| Sliding Window Attention |:x:|:x:|
|**Encoders**| | |
| VQ-VAE |:x:|:x:|
| VAE |:x:|:heavy_check_mark:|
| CLIP |:x:|:heavy_check_mark:|
|**Schedulers**| | |
| DDPM |:x:|:heavy_check_mark:|
|**Search Algorithms**| | |
| Monte Carlo Tree Search |:x:|:x:|
