"""Qwen Image 2.1 RefMod V3: loss-minimized, reusable pixel + VAE-latent reference profile.

The Qwen visual encoder still tokenizes the reference images at generation time;
only the VAE reference latents are cached. Not a trained character embedding.
Based on the public TextEncodeQwenImage21 implementation in ComfyUI.
"""
import os
import re
import math
import json
import torch
import comfy.utils
import comfy.model_management
import node_helpers
import folder_paths
from safetensors.torch import save_file, load_file
from safetensors import safe_open

_BASE = os.path.join(folder_paths.get_input_directory(), 'qwen_refmods_v3')
os.makedirs(_BASE, exist_ok=True)

def safe_name(value):
    value=str(value).strip()
    if not re.fullmatch(r'[\w .-]{1,100}',value,flags=re.UNICODE) or value in ('.','..'):
        raise ValueError('Nombre invalido. Usa letras, numeros, espacio, punto, _ o - (sin rutas).')
    return value[:-12] if value.endswith('.safetensors') else value

def resize21(image,resolution):
    if image is None or not isinstance(image,torch.Tensor) or image.ndim!=4:
        raise ValueError('Se requiere tensor IMAGE de ComfyUI [B,H,W,C]')
    samples=image[:1].movedim(-1,1)
    h,w=samples.shape[2:4]
    if resolution>0:
        ratio=w/h
        width=round(math.sqrt(resolution*resolution*ratio)/32)*32
        height=round(math.sqrt(resolution*resolution/ratio)/32)*32
    else:
        width,height=round(w/32)*32,round(h/32)*32
    width,height=max(32,width),max(32,height)
    if (width,height)==(w,h):
        return image[:1]
    return comfy.utils.common_upscale(samples,width,height,'lanczos','disabled').movedim(1,-1)

def vision_rgb(s):
    rgb=s[:,:,:,:3]
    if s.shape[-1]>3:
        a=s[:,:,:,3:4]
        rgb=rgb*a+(1-a)
    return rgb

def pack_preview(t):
    return t.detach().to('cpu',torch.float16).contiguous()

class Qwen21RefModCreate8V3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'profile_name':('STRING',{'default':'Personaje_01'}),
                            'vae':('VAE',),
                            'image_1':('IMAGE',),
                            'resolution':('INT',{'default':768,'min':256,'max':2048,'step':32}),
                            'profile_description':('STRING',{'default':'Una sola persona adulta; identidad consistente','multiline':True})},
                'optional':{f'image_{i}':('IMAGE',) for i in range(2,9)}}
    RETURN_TYPES=('STRING','STRING','IMAGE')
    RETURN_NAMES=('profile_filename','details','first_view')
    FUNCTION='create'
    CATEGORY='Qwen RefMod V3'
    OUTPUT_NODE=True
    def create(self,profile_name,vae,image_1,resolution,profile_description,**kwargs):
        name=safe_name(profile_name)
        images=[image_1]+[kwargs.get(f'image_{i}') for i in range(2,9)]
        images=[i for i in images if i is not None]
        if len(images)>8: raise ValueError('Maximo ocho referencias')
        tensors={}
        for idx,img in enumerate(images,1):
            resized=resize21(img,resolution)
            rgb=vision_rgb(resized)
            # Qwen21 official: vision tower sees RGB over white; VAE may receive RGBA.
            latent=vae.encode(resized)
            tensors[f'image_{idx}']=pack_preview(rgb)
            tensors[f'latent_{idx}']=latent.detach().to('cpu',torch.float32).contiguous()
        metadata={'format':'qwen21_refmod_v3','views':str(len(images)),
                  'resolution':str(resolution),'description':profile_description,
                  'warning':'VAE latents tied to the Qwen Image 2.1 VAE used during extraction. Vision images still processed each generation.'}
        filename=name+'.safetensors'
        target=os.path.join(_BASE,filename)
        save_file(tensors,target,metadata=metadata)
        return (filename,f'Guardado {filename} con {len(images)} imagenes RGB + {len(images)} latentes VAE a {resolution}px. Ruta: {target}',vision_rgb(resize21(images[0],resolution)))


def _soften_reference(tensor, strength):
    """Blend reference toward its own spatial low-pass version, not zeros/noise.

    ComfyUI IMAGE is BHWC; Qwen VAE latent is BCHW. References are in CPU
    storage. This mechanism is inspired by H3 ref_block; Qwen uses a distinct
    conditioning architecture, so equivalent perceptual strength is NOT guaranteed.
    """
    import torch.nn.functional as F
    if strength >= 1.0:
        return tensor
    is_image = tensor.ndim == 4 and tensor.shape[-1] in (3,4)
    x = tensor.movedim(-1,1) if is_image else tensor
    if x.ndim != 4:
        raise ValueError('Unexpected reference dimensions')
    h,w = x.shape[-2:]
    sh,sw=max(1,h//16),max(1,w//16)
    blurred=F.interpolate(F.adaptive_avg_pool2d(x,(sh,sw)),size=(h,w),mode='bilinear',align_corners=False)
    mixed=x*strength+blurred*(1.0-strength)
    return mixed.movedim(1,-1).contiguous() if is_image else mixed.contiguous()

class Qwen21RefModTextToImageV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'clip':('CLIP',),
            'profile_filename':('STRING',{'default':'Personaje_01.safetensors'}),
            'prompt':('STRING',{'multiline':True,'default':'Full-body cinematic portrait of the same person, standing in an elegant garden, natural light, photorealistic.'}),
            'negative_prompt':('STRING',{'multiline':True,'default':'extra people, different face, distorted anatomy, collage, text, watermark'}),
            'views_to_use':('INT',{'default':8,'min':1,'max':8}),
            'strength':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05}),
            'copies':('INT',{'default':1,'min':1,'max':10}),
            'retention':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.05}),
            'width':('INT',{'default':1024,'min':256,'max':2048,'step':32}),
            'height':('INT',{'default':1024,'min':256,'max':2048,'step':32}),
            'identity_instruction':('STRING',{'multiline':True,'default':'The reference pictures show the SAME ONE adult person from different angles. Preserve the exact identity, facial geometry, distinctive features, skin tone and natural body proportions. Change the scene, clothing and pose according to the request. One person only.'}),
        }}
    RETURN_TYPES=('CONDITIONING','CONDITIONING','LATENT','STRING')
    RETURN_NAMES=('positive','negative','latent','profile_details')
    FUNCTION='encode'
    CATEGORY='Qwen RefMod V3'
    def encode(self,clip,profile_filename,prompt,negative_prompt,views_to_use,strength,copies,retention,width,height,identity_instruction):
        name=safe_name(profile_filename)+'.safetensors'
        path=os.path.join(_BASE,name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'No existe el perfil {path}. Ejecuta CREATE REFMOD primero.')
        with safe_open(path,framework='pt',device='cpu') as f:
            metadata=f.metadata() or {}
            if metadata.get('format')!='qwen21_refmod_v3':
                raise ValueError('Formato incompatible: se requiere perfil Qwen 2.1 RefMod V3.')
            count=min(int(metadata['views']),views_to_use)
            images=[f.get_tensor(f'image_{i}') for i in range(1,count+1)]
            latents=[f.get_tensor(f'latent_{i}') for i in range(1,count+1)]
        # MiniMax RefMod-inspired controls: repeat the complete profile, not a selected anchor.
        # Both VL image tokens and Qwen VAE reference latents must be attenuated.
        # This is an experimental analogue, not an exact transfer of H3's PackedLayout.
        effective = max(0.0, min(1.0, float(strength) * float(retention)))
        if count * copies > 16 and effective > 0:
            raise ValueError(f'{count} vistas x {copies} copias = {count*copies} referencias; Qwen Image 2.1 soporta hasta 16. Reduce copias o vistas.')
        if effective == 0:
            images_vl=[]
            ref_latents=[]
        else:
            images_vl=[]
            ref_latents=[]
            for _ in range(copies):
                for img,latent in zip(images,latents):
                    images_vl.append(_soften_reference(img.float(),effective))
                    ref_latents.append(_soften_reference(latent.float(),effective))
        positive_text=identity_instruction.strip()+'\n'+prompt.strip()
        # Match the TextEncodeQwenImage21 path with VAE reference latents present.
        positive=clip.encode_from_tokens_scheduled(clip.tokenize(positive_text,images=images_vl if images_vl else None,keep_vision=False,prevent_empty_text=True))
        negative=clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt,images=images_vl if images_vl else None,keep_vision=False,prevent_empty_text=True))
        if ref_latents:
            positive=node_helpers.conditioning_set_values(positive,{'reference_latents':ref_latents},append=True)
            negative=node_helpers.conditioning_set_values(negative,{'reference_latents':ref_latents},append=True)
        # Qwen Image 2.1: 64-channel latent, 16x spatial downscale.
        empty=torch.zeros([1,64,height//16,width//16],device=comfy.model_management.intermediate_device())
        details=f'Perfil: {name}; vistas: {count}; copias: {copies}; fuerza: {strength:.2f}; retencion: {retention:.2f}; fuerza efectiva: {effective:.2f}; referencias: {len(ref_latents)}; cache: {metadata.get("resolution")}; salida: {width}x{height}.'
        return (positive,negative,{'samples':empty},details)

NODE_CLASS_MAPPINGS={
    'Qwen21RefModCreate8V3':Qwen21RefModCreate8V3,
    'Qwen21RefModTextToImageV3':Qwen21RefModTextToImageV3
}
NODE_DISPLAY_NAME_MAPPINGS={
    'Qwen21RefModCreate8V3':'QWEN 2.1 REFMOD V3 | 01 CREATE 8 VIEWS',
    'Qwen21RefModTextToImageV3':'QWEN 2.1 REFMOD V3 | 02 STRENGTH + COPIES + RETENTION'
}
