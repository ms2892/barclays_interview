# Barclays Interview

## Overview

This problem is a multilabel classficiation problem where we are given titles and summaries of various research papers and the task is to identify the labels that would be associated with the respective paper. Also we are provided the taxonomy used to classify these different research papers. 

From the taxonomy it can be deduced that it is a 2 level hierarchical taxonomy and something like a hierarchical based multilabel classification model could work well here. To keep the scope of the project simpler, the 

## Install Requirements

Make sure to create a virtual enviroment before installing the requirements.

```
python -m venv .venv
source .venv/bin/activate
.venv\Scripts\activate
```

## Prepare the Dataset

To prepare the dataset to the format that would be needed for the training script

```
python prepare_dataset.py --taxonomy category_taxonomy.csv --dataset arxiv_data.csv --output_file data_cleaned.csv
```

## Training Scripts

### To train using only the title

```
python training_script_title.py --clean_dataset data_cleaned.csv  
```

### To train using title and summaries

```
python training_script_summary.py --clean_dataset data_cleaned.csv  
```

These training scripts will save the model in the models folder in an ONNX Format

The training config can be used to define the triton inference server that can be used

## Triton Inference Server

To run the inference server use the following command. Before that make sure that the onnx_model folder structure looks something like this

```
onnx_models/
└── title_model/
    ├── 1/
    │   └── title_model.onnx
    └── config.pbtxt

```

Then run the following docker command to start a triton inference server

```
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/onnx_models:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models
```