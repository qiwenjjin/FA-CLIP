# FA-CLIP (partial release)

Fine-Grained Aligned CLIP for Zero-Shot Anomaly Detection.

Partial source for the TCSVT response letter: implementation details that the letter
points to GitHub for (batch size, prompt templates, cache / ITE / ODA settings,
frozen inference). Encoders, datasets, and end-to-end scripts are omitted.

## Layout

```
train.py                 # CLI defaults aligned with the manuscript
method/
  prompts.py             # Literal normal/abnormal prompt templates
  oda.py                 # ODALoss (Sec. III-C)
  cache.py               # MemoryCache (ITE text) + ImageMemoryCache
  faclip.py              # FaCLIP: dual-cache refine + L_ODA
  loss.py                # Focal / Dice for L_cls and L_seg (Eq. 23)
  trainer.py             # L_total = L_cls + L_seg + λ_ODA L_ODA (Eq. 22)
  __init__.py
```



## Manuscript defaults

| Item | Default |
|------|---------|
| Backbone | ViT-L-14-336 (encoder omitted here) |
| Feature dim | 768 |
| Image size | 518 |
| Batch size | 1 |
| Epochs / lr | 5 / 0.01 |
| Optimizer | AdamW |
| Seeds | 111 (also 222, 333) |
| Cache `M`, `m`, `η` | 25, 0.8, 0.2 |
| `N_warmup`, `N_thres` | 50, 100 |
| `λ_ODA` | 0.05 |
| `γ` init | −1 (learnable) |
| Text-cache gate | ITE |
| Default sources | MVTec AD + ColonDB (`--zsad_protocol swap` → VisA + ClinicDB) |
