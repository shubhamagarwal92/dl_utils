
## TL;DR

InternVL (v2→v3) — “V2PE” (Variable Visual Position Encoding)

Instead of treating visual tokens like text, InternVL shrinks the positional step for visual tokens (denser positions) so long multimodal contexts don’t blow up RoPE’s effective span. V2PE is slotted into InternVL3 to extend multimodal context windows; conceptually it’s compatible with RoPE by remapping indices for images/videos. 

SigLIP2 (Google) — ViT with learned absolute PEs (not RoPE)

SigLIP2 sticks to ViT-style learned absolute position embeddings for both image and text encoders; no rotary. This holds across FixRes and NaFlex variants. 

Recent research threads

Circle-RoPE decouples text and image token manifolds (text along a line, image tokens on a circle/cone) to reduce cross-modal bias from shared 1D indices. Good mental model for why decomposed/decoupled RoPE helps. 

Qwen2-VL / Qwen2.5-VL — “M-RoPE” (decomposed 1D/2D/3D)

They separate positions into text 1D and visual 2D/3D (height, width, and optional time for video) and apply RoPE separately to each axis before fusing, so an image patch’s rotation uses (t, h, w) instead of a big flat 1D index. This avoids text–image index entanglement and scales to video. 



