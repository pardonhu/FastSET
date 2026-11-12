# Model card

FastSET uses a neural channel estimator and a neural bit demapper. The estimator
adds a learned residual to a linearly interpolated pilot LS estimate. It is
trained with true-channel MSE. The demapper predicts coded-bit logits and is
trained with BCE. True channel labels supervise training and are not a receiver
inference input. Both modules consume the same OFDM grid/noise conventions as the
reference implementation.

`models/grid.py` contains the original CNN model definitions and spindle channel
rule. A split ID records estimator residual-block depth and initial filter width.
The demapper is allocated from the remaining 661,188-parameter E2E budget.
`fastset check --config ...` reports the exact architecture and parameter count.
All measured depths1–10 and widths8/16/32/64 have explicit configurations.

The Transformer and MLP definitions are in `backbones/`. Configurations include
all depth, filter-width, attention-head and token-hidden dimensions, and the
parameter budget is checked before execution. The internal name `mixer` identifies
the released MLP implementation; paper legends call it MLP.

Main models use source estimator5/demapper10 epochs or E2E10 epochs. The source
checkpoint is selected by the fixed source budget. The Release bundle includes
source-pretrained states with zero online Adam counters, plus separately labeled
final target-adapted models for independent evaluation. Those final models must
not initialize a new online comparison.

Sequential online training updates only the active module. Frozen model weights
are hashed before and after each update. Joint SP reuses sequential source
weights and updates both modules with MSE+BCE. Joint JP uses joint source
pretraining and the same joint online objective. Each loss has coefficient one.
The 30-epoch E2E control is provided as a from-scratch configuration; its source
state is not included in the model archive.

The source-model difference is explicit in experiment IDs and legends.

The package exposes both source pretraining and online adaptation. All cases
reuse shared model, data, optimizer and LDPC functions. Neural execution requires
a GPU; input construction and the Sionna LMMSE linear-algebra workaround can run
on CPU. Environment and device records are saved for every run.
