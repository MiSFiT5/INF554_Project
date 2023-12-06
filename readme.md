# INF554 - Machine and Deep Learning Data Challenge
Extractive Summarization with Discourse Graphs

## Notes
This program is implemented base on Windows 10 x64 System. Using Python 3.8 and PyTorch with CUDA 12.1. Update to newer version of PyTorch using instructions from [PyTorch Install Guide](https://pytorch.org/get-started/locally/)

## Requirements

Make sure you have the required dependencies installed by running:
```bash
pip install -r requirements.txt
```

## Command-line Arguments

- `--train_path`: Specify the path to the training data. (Default: None)
- `--test_path`: Specify the path to the test data. (Default: None)
- `--labels_path`: Specify the path to the labels. (Default: None)
- `--model`: Specify the model to use. (Default: DecisionTree)
- `--result_path`: Specify the path to save the result. (Default: test_labels.json)


## Running

Use the following command to run the main script:

```bash
python main.py --train_path /path/to/training_data --test_path /path/to/test_data --labels_path /path/to/labels --model YourModel --result_path /path/to/result_file.json
```

Replace `/path/to/training_data`, `/path/to/test_data`, `/path/to/labels`, `YourModel`, and `/path/to/result_file.json` with your actual paths and model choice.

## Generate Result

In the end, convert the results in json format to kaggle-compatible submission.csv with:

```bash
python make_submission.py --json_path /path/to/result_file.json
```
Replace `/path/to/result_file.json` with your actual paths.

## Supported Models

- DecisionTree
- RandomForest
- XGBoost
- TwoModels
- LSTM
- GCN
- GAT
- GraphSAGE
