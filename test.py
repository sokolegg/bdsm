from loss_functions import *
from bdsm import BDSMPipeline
from diffusers import LCMScheduler


def test_anchor_latents():
    anchor = torch.ones([1, 4, 64, 64])
    distance_f = euclidean_distance_between_anchor_latents(anchor)

    model_id = "Lykon/dreamshaper-7"
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = BDSMPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    prompt = "Cat is dancing"

    # disable guidance_scale by passing 0
    image = pipe(prompt=prompt,
                 num_inference_steps=4,
                 guidance_scale=0,
                 loss_function=distance_f,
                 bm_width=10,
                 bdsm_prompt_embeds_scale=0.1,
                 ).images[0]

    img_name = "images/test_anchor_latents.jpg"
    image.save(img_name)


if __name__ == "__main__":
    test_anchor_latents()