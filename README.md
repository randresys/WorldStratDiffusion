# WorldStratDiffusion

Lines of code for executing training:

```bash
!accelerate launch training_script.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="/content/mixed_output" --mask_data_dir="/content/mixed_input" --class_data_dir="training_data/satelitals" --output_dir="stable_diffusion_weights/satelitals" --instance_prompt="" --class_prompt="" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --use_8bit_adam --enable_xformers_memory_efficient_attention --learning_rate=2e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=2500 --num_class_images=8
```
Líneas de código obtener resultados del modelo obtenido:
```python
from diffusers import StableDiffusionPipeline
import torch
from torchvision import transforms

transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

pipe = StableDiffusionPipeline.from_pretrained('/kaggle/working/stable_diffusion_weights/satelitals4',
                         torch_dtype=torch.float16, revision="fp16",
                         safety_checker = None,requires_safety_checker = False)
pipe = pipe.to("cuda")

```
