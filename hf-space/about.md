### About

Braid is a research LM architecture by
[Solexsis Research](https://solexsis.com).
It replaces most of a transformer's attention with gated causal convolutions
inside a hierarchical U-Net, spending attention only where global context
is needed. At 30M scale that measured 3.07x the training throughput and 2.23x less peak
memory than a matched transformer, at a 0.019 bpb quality cost, with no
tokenizer.
This 300M checkpoint was trained on 128 GB of openly licensed text
(the open-v1 mixture; each source keeps its own open licence). Weights and
runtime are Apache-2.0.
Source: [github.com/Solexsis/braid-demo](https://github.com/Solexsis/braid-demo).
