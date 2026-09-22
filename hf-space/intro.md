# Braid

**Braid 300M** is a tokenizer-free byte-level language model -- 295M parameters,
vocabulary of 256 raw byte values, no tokenizer anywhere. It is a causal U-Net
(HourglassBraid) that runs dense attention at 1/16 of positions and spends the
rest on cheap gated convolutions. This is a **research preview**: the
architecture trades quality for systems cost, and the model is a base
text-continuation model, not a chatbot. Write the start of something (a
sentence, a paragraph, a code block) and it will continue it.
Weights and the openly licensed training corpus are at
[Solenopsisbot/braid-300m](https://huggingface.co/Solenopsisbot/braid-300m) and
[Solenopsisbot/braid-open-v1](https://huggingface.co/datasets/Solenopsisbot/braid-open-v1).
