# Qwen Image 2.1 RefMod V3 — adaptación a partir del RefMod original de MiniMax H3

## Incluye
- `01_CREAR_REFMOD_8_FOTOS.json`: ocho entradas LoadImage, VAE Qwen, guarda `.safetensors` con RGB y latentes de referencia.
- `02_GENERAR_IMAGEN_CON_REFMOD.json`: generación texto a imagen reutilizando el perfil y controles `strength` (0–1), `copies` (1–10) y `retention` (0–1).
- `__init__.py`: nodos personalizados para ComfyUI.

## Instalación
1. Descomprimir y copiar la carpeta `QwenRefModV3` a `ComfyUI/custom_nodes/`.
2. Reiniciar ComfyUI y verificar que los nodos aparezcan.
3. Cargar el workflow 01, seleccionar 1 a 8 fotos de la misma persona, configurar `profile_name`, ejecutar.
4. Cargar el workflow 02, especificar el nombre del perfil `.safetensors`, escribir el prompt, ejecutar.
5. Los perfiles se almacenan en `ComfyUI/input/qwen_refmods_v3/`.

## Controladores (basados en comportamiento comprobado del código MiniMax)
- **strength 0–1**: atenua imágenes de referencia y latentes VAE hacia versiones suavizadas. Con 0, no envía referencias.
- **copies 1–10**: repite la secuencia completa de vistas, no una foto principal. El número total de referencias no puede superar **16** (límite de Qwen 2.1). Ejemplo: 8 vistas => máximo 2 copias.
- **retention 0–1**: multiplicador global: `strength * retention` (limitado a 0–1).
- **views_to_use 1–8**: número de vistas utilizadas.

## Límites y observaciones
- La escala de fuerza es **experimental**, no está calibrada ni significa idéntico efecto al RefMod de H3. La ruta de Qwen no implementa `PackedLayout` / `minimax_refs`.
- En el MiniMax original, el decaimiento se aplica en `H3RefMod.ref_block` con mezcla de latente y copia borrosa, y `MiniMaxH3RefModsLoader` duplica los bloques; en Qwen adaptamos la idea para el camino **Qwen3-VL + latentes VAE**.
- El perfil no es un embedding facial entrenado: conserva RGB más latentes VAE; el codificador Qwen3-VL debe procesar imágenes en cada generación.
- Las copias repetidas consumen más VRAM, y `8 vistas x 3 copias` produce un error explícito, no un recorte silencioso.
- Mantener el mismo VAE para crear y generar. No hay prueba real en GPU, solo validación estática de sintaxis y enlaces.
- Para referencias autorizadas / imágenes propias.

## Referencias
- Código aportado: `ComfyUI-MiniMaxH3Mod-main.zip`, archivos `core.py` (H3RefMod.ref_block) y `nodes.py` (MiniMaxH3RefModsLoader, _ref_blocks, MiniMaxH3RefModApply).
- ComfyUI: `comfy_extras/nodes_qwen.py` `TextEncodeQwenImage21`.
