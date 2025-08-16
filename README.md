# Text-to-Image Generation with Hugging Face Diffusers

**Notebook:** `GenerativeImagesHuggingFace.ipynb`

This project demonstrates **text-to-image generation** using Hugging Face **Diffusers**. It shows how to load public diffusion models, configure schedulers, control guidance/steps/seeds, and optionally use **negative prompts**. You’ll generate images reproducibly and save them to disk, with optional GPU acceleration and mixed precision.

---

## 🚀 What you’ll build
- Load a Stable Diffusion checkpoint from **Hugging Face Hub**
- Create a **Diffusers pipeline** (e.g., StableDiffusionPipeline)
- Configure a scheduler (EulerDiscreteScheduler, DDIMScheduler, PNDMScheduler)
- Tweak **`guidance_scale`**, **`num_inference_steps`**, image **`height`/`width`**, and **seed**
- Generate images for different prompts (and **negative_prompt** if used)
- Save results to disk (e.g., `images/*.png`)

---

## 🧰 Environment & Setup

**Dependencies:** `diffusers`, `transformers`, `accelerate`, `safetensors`, `torch`, `xformers` (optional)

Install:
```bash
pip install -U torch diffusers transformers accelerate safetensors
# optional speedups:
pip install -U xformers
```

**Authentication:** (if a gated model is used)
```bash
export HUGGINGFACEHUB_API_TOKEN=<your_token>
```
*(Token use detected: No)*

**Device & dtype:** CUDA: Yes · FP16: Yes · Safety checker disabled: No

---

## 🧠 Models & Pipelines

**Models referenced:**
- docs/diffusers
- runwayml/stable-diffusion-v1-5
- stabilityai/stable-diffusion-2-1-base

**Pipelines detected:** StableDiffusionPipeline  
**Schedulers detected:** EulerDiscreteScheduler, DDIMScheduler, PNDMScheduler

---

## 🎛️ Generation Controls

- `guidance_scale`: 7.5, 7.5, 7.5  
- `num_inference_steps`: 30, 30, 30  
- `height`×`width`: 512×512  
- `seed`: set via `torch.manual_seed(<int>)`  
- Negative prompt used: No

**Prompt examples:** An astronaut taking a selfie in space; An astronaut taking a selfie in space

**Negative prompt example:** optional: things to avoid

---

## 🖼️ Saving Outputs

- images will be saved where you call `.save(...)`

Create a folder to organize results:
```bash
mkdir -p images
```

---

## 📁 Repository Structure
```text
├── GenerativeImagesHuggingFace.ipynb
├── images/                  # generated images (add .gitignore if large)
└── README.md
```

---

## ✅ Usage (Minimal Example)
```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "docs/diffusers"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a cinematic photo of a vintage motorcycle at sunset"
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30, generator=torch.manual_seed(42)).images[0]
image.save("images/sunset_bike.png")
```

---

## 🔒 Notes on Safety & Licensing
- Respect each model’s **license** and **usage policy** on its Hugging Face model card.
- The safety checker can filter NSFW content; disabling it may surface unwanted generations. (Detected disabled: No.)

---

## 🧭 Next Steps
- Try **SDXL** and compare results/VRAM use.  
- Experiment with **img2img** and **inpainting** pipelines.  
- Switch schedulers (Euler A / UniPC / DDIM) and compare speed vs quality.  
- Use **LoRA** or **ControlNet** for style or structure control.
