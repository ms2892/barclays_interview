# barclays_interview

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

To run the inference server use the following command

```
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:23.04-py3 \
  tritonserver --model-repository=/models
```