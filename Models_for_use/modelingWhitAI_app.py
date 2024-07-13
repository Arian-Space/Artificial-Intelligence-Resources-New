import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    st.write("Cargando modelos...")
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    return xm, model, diffusion

def generate_latents(model, diffusion, prompt, batch_size=4):
    st.write(f"Generando latents para: {prompt}")
    guidance_scale = 15.0

    return sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

def decode_and_render(xm, latents):
    st.write("Decodificando y renderizando...")
    render_mode = 'nerf'
    size = 64

    cameras = create_pan_cameras(size, device)
    all_images = []
    for latent in latents:
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        all_images.append(images[0].cpu().numpy())  # Tomamos solo el primer frame y lo convertimos a numpy array
    return all_images

def main():
    st.title("Generador de Modelos 3D")

    xm, model, diffusion = load_models()

    prompt = st.text_input("¿Qué quieres modelar?", "una silla")

    if st.button("Generar"):
        with st.spinner("Generando modelos 3D..."):
            latents = generate_latents(model, diffusion, prompt, batch_size=4)
            images = decode_and_render(xm, latents)

        cols = st.columns(4)
        for i, img in enumerate(images):
            img_normalized = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)
            with cols[i]:
                st.image(img_pil, caption=f"Modelo {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()