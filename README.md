# FA-CLIP (partial release)

Fine-Grained Aligned CLIP for Zero-Shot Anomaly Detection.

Partial source for the TCSVT response letter: implementation details that the letter
points to GitHub for (batch size, prompt templates, cache / ITE / ODA settings,
frozen inference). Encoders, datasets, and end-to-end scripts are omitted.

## Layout

```
train.py                 # CLI defaults aligned with the manuscript
prompts.py             # Literal normal/abnormal prompt templates
oda.py                 # ODALoss (Sec. III-C)
cache.py               # MemoryCache (ITE text) + ImageMemoryCache
faclip.py              # FaCLIP: dual-cache refine + L_ODA
loss.py                # Focal / Dice for L_cls and L_seg (Eq. 23)
trainer.py             # L_total = L_cls + L_seg + λ_ODA L_ODA (Eq. 22)
__init__.py
```
