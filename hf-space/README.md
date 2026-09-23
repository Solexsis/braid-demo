---
title: Braid
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Tokenizer-free byte-level hourglass LM (research preview)
models:
  - Solexsis/braid-300m
datasets:
  - Solenopsisbot/braid-open-v1
---

# Braid demo (ZeroGPU)

Source for the `Solenopsisbot/braid` Space. It contains no model code: the
weights package and the matching runtime wheel both come from the model repo
at startup. The source of truth is `hf-space/` in
https://github.com/Solexsis/braid-demo.
