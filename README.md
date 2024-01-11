# OVLA on GLUE Benchmark

This repository is dedicated to the application of OVLA within the GLUE benchmark tasks, featuring our tailored implementation of the OVLA-enhanced BERT model, alongside a specialized trainer tailored for the NLP tasks encompassed by the GLUE benchmark. The accompanying table provides a detailed overview of the performance outcomes.

| Dataset | AC | AW | ACU | AWU |
| :--- | :---: | :---: | :---: | :---: |
|  | $83.60 \%$ | $27.13 \%$ | - | - |
| CoLA | $83.41 \%$ | $24.35 \%$ | $83.32 \%$ | $100.0 \%$ |
|  | $91.17 \%$ | $47.82 \%$ | - | - |
| SST-2 | $91.51 \%$ | $48.85 \%$ | $91.51 \%$ | $100.0 \%$ |
|  | $83.58 \%$ | $26.96 \%$ | - | - |
| MRPC | $85.53 \%$ | $26.47 \%$ | $86.27 \%$ | $100.0 \%$ |
|  | $66.79 \%$ | $56.68 \%$ | - | - |
| RTE | $66.79 \%$ | $67.51 \%$ | $67.51 \%$ | $100.0 \%$ |
|  | $90.83 \%$ | $50.28 \%$ | - | - |
| QNLI | $90.68 \%$ | $50.32 \%$ | $90.61 \%$ | $99.86 \%$ |
|  | $90.80 \%$ | $62.05 \%$ | - | - |
| QQP | $90.88 \%$ | $62.56 \%$ | $90.88 \%$ | $100.0 \%$ |
|  | $83.86 \%$ | $33.68 \%$ | - | - |
| MNLI | $83.11 \%$ | $32.74 \%$ | $83.05 \%$ | $99.82 \%$ |
|  | $84.01 \%$ | $34.76 \%$ | - | - |
| MNLI-mm | $83.54 \%$ | $33.93 \%$ | $83.60 \%$ | $99.78 \%$ |

The OVLA watermarked-BERT model, assessed using the Hugging Face framework, involves creating a specialized dataset integrating watermarked and clean data. A tailored BERT model for sequence classification processes this dataset, outputting four predictions (AC, AW, ACU, AWU). Unique watermark keys cater to tasks with varying label class numbers. Training parameters are matched with those of standard BERT to ensure consistency and fair comparison. Post-training, the model's perturbed head is pruned for practical application.

To explore the watermarked datasets used by our model, please refer to the [OVLA_GLUE dataset](https://huggingface.co/datasets/simon508/OVLA_GLUE) on Hugging Face. Additionally, the fully trained OVLA-BERT model is available for download and further experimentation at the [OVLA_GLUE_BERT model repository](https://huggingface.co/simon508/OVLA_GLUE_BERT).
