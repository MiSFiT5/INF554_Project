from utils import *
from models import *
from process import *
from pathlib import Path
import json
from argparse import ArgumentParser


def main():
    # load paths
    print("Loading paths...")
    path_to_training = Path(args.train_path)
    path_to_test = Path(args.test_path)
    path_to_labels = Path(args.labels_path)
    
    # load dataset
    print("Loading dataset...")
    training_set, validate_set, test_set = split_dataset(validate=True)
    
    # train model and predict
    print("Training model and predicting...")
    if args.model == "DecisionTree" or args.model == "RandomForest" or args.model == "XGBoost":
        test_labels = process_Tree(args.model, training_set, validate_set, test_set, path_to_training, path_to_training, path_to_test, path_to_labels)
    elif args.model == "TwoModels":
        test_labels = process_twomodels(training_set, validate_set, test_set, path_to_training, path_to_training, path_to_test, path_to_labels)
    elif args.model == "LSTM":
        test_labels = process_LSTM(training_set, validate_set, test_set, path_to_training, path_to_training, path_to_test, path_to_labels)
    elif args.model == "GCN" or args.model == "GAT" or args.model == "GraphSAGE":
        test_labels = process_GNN(args.model, training_set, validate_set, test_set, path_to_training, path_to_training, path_to_test, path_to_labels)
    else:
        raise ValueError("Invalid model")
    
    # save result
    print("Saving result...")
    with open(args.result_path, "w") as file:
        json.dump(test_labels, file, indent=4)
    
    # done
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str,
                        help="specify the path to the training data",
                        default=None)
    parser.add_argument("--test_path", type=str,
                        help="specify the path to the test data",
                        default=None)
    parser.add_argument("--labels_path", type=str,
                        help="specify the path to the labels",
                        default=None)
    parser.add_argument("--model", type=str,
                        help="specify the model to use",
                        default="DecisionTree")
    parser.add_argument("--result_path", type=str,
                        help="specify the path to the result",
                        default="test_labels.json")
    args = parser.parse_args()
    
    main()
    