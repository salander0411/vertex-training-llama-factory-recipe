## LLaMA-Factory Multi-Node Recipes on Vertex Custom Training 

This doc provides step-by-step guidance on how to use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to do multi-node fine-tuning on Vertex Custom Training, taking an example model of [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct). Check [this link](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models) for a full list of supported models on LLaMA-Factory. 

1. [H200 recipe](https://github.com/salander0411/vertex-training-llama-factory-recipe/tree/main/a3-ultra): Using [a3-ultragpu-8g](https://cloud.google.com/compute/docs/gpus#h200-gpus) & DWS Reservation capacity type. 
2. [H100 recipe](https://github.com/salander0411/vertex-training-llama-factory-recipe/tree/main/a3-mega):  Using [a3-megagpu-8g (H100 mega)](https://cloud.google.com/compute/docs/gpus#h100-gpus) & spot capacity type.

## Prerequisite 
1. Familiarity with LLaMA-Factory. Refer to [the documentation](https://llamafactory.readthedocs.io/en/latest/) for comprehensive information. Also, understanding of Distributed Training and model parallelism settings. Consult [this page](https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html#) for additional details.
2. Enable APIs: Verify that the Vertex AI API is enabled for your project.
