
from utils import *
from models import *
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNNDataLoader
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

# Load pre-trained SentenceTransformer model
bert = SentenceTransformer('all-MiniLM-L6-v2')


def process_Tree(model_name, training_set, validate_set, test_set, path_to_training, path_to_validate, path_to_test, path_to_labels):
    # Get text features and labels for the training set
    X_training = get_text_feature(training_set, path_to_training)
    y_training = get_label(training_set, path_to_labels)

    # Get text features and labels for the validation set
    X_validate = get_text_feature(validate_set, path_to_validate, show_progress_bar=False)
    y_validate = get_label(validate_set, path_to_labels)
    
    # Initialize the classifier based on the specified model_name
    if model_name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=0)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, criterion='gini', n_jobs=-1, random_state=0)
    elif model_name == "XGBoost":
        clf = XGBClassifier(n_estimators=100, max_depth=25, objective='binary:logistic', n_jobs=-1, random_state=0)
    else:
        raise ValueError("model_name must be one of 'DecisionTree', 'RandomForest' and 'XGBoost'.")
    
    # Train the classifier on the training set
    clf.fit(X_training, y_training)

    # Predict labels for the validation set
    y_pred = clf.predict(X_validate).tolist()

    # Print F1 score for the validation set
    f1 = f1_score(y_validate, y_pred, average='binary')
    print(f'F1 score of {model_name}: {f1:.4f}')
    
    # Predict labels for the test set
    test_labels = {}
    for transcription_id in test_set:
        with open(path_to_test / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
        
        X_test = []
        for utterance in transcription:
            X_test.append(utterance["speaker"] + ": " + utterance["text"])
        
        X_test = bert.encode(X_test)

        y_test = clf.predict(X_test)
        test_labels[transcription_id] = y_test.tolist()
    
    return test_labels


def process_LSTM(training_set, validate_set, test_set, path_to_training, path_to_validate, path_to_test, path_to_labels):
    # Get text features and labels for the training set
    X_training = get_text_feature(training_set, path_to_training)
    y_training = get_label(training_set, path_to_labels)

    # Get text features and labels for the validation set
    X_validate = get_text_feature(validate_set, path_to_validate, show_progress_bar=False)
    y_validate = get_label(validate_set, path_to_labels)
    
    # Define model hyperparameters
    input_size = X_training.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1  # Binary classification task

    # Initialize the LSTM model
    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to PyTorch tensors
    X_training_tensor = torch.tensor(X_training)
    y_training_tensor = torch.tensor(y_training, dtype=torch.int)
    X_validate_tensor = torch.tensor(X_validate)
    y_validate_tensor = torch.tensor(y_validate, dtype=torch.int)

    # Create TensorDataset
    train_dataset = TensorDataset(X_training_tensor, y_training_tensor)
    validate_dataset = TensorDataset(X_validate_tensor, y_validate_tensor)

    # Use DataLoader to load data
    batch_size = 64  # Adjust batch size as needed
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

    # Model training
    num_epochs = 25
    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        
        # Model evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in validate_dataloader:
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                # Predict
                outputs = model(inputs)
                predictions = (outputs >= 0.5).int()
                
                # Save predictions and labels
                all_predictions.extend(predictions.numpy())
                all_labels.extend(labels.numpy())

        # Calculate f1-score
        f1 = f1_score(all_labels, all_predictions)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, F1-Score on validation set: {f1:.4f}')
    
    # Predict test set
    test_labels = {}
    model.eval()
    with torch.no_grad():
        for transcription_id in test_set:
            with open(path_to_test / f"{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            
            X_test = []
            for utterance in transcription:
                X_test.append(utterance["speaker"] + ": " + utterance["text"])
            
            X_test = bert.encode(X_test)
            X_test = torch.tensor(X_test).unsqueeze(1)

            outputs = model(X_test)
            y_test = (outputs >= 0.5).int()
            y_test = y_test.squeeze(1)
            test_labels[transcription_id] = y_test.tolist()
    
    return test_labels


def process_twomodels(training_set, validate_set, test_set, path_to_training, path_to_validate, path_to_test, path_to_labels):
    # Get text features, graph features, and labels for the training set
    text_feature_training = get_text_feature(training_set, path_to_training)
    graph_feature_training, relation_mapping = get_graph_feature(training_set, path_to_training)
    y_training = get_label(training_set, path_to_labels)

    # Get text features, graph features, and labels for the validation set
    text_feature_validate = get_text_feature(validate_set, path_to_validate, show_progress_bar=False)
    graph_feature_validate, _ = get_graph_feature(validate_set, path_to_validate, relation_mapping)
    y_validate = get_label(validate_set, path_to_labels)
    
    # Use LSTM for text feature
    # Define model hyperparameters
    input_size = text_feature_training.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1  # Binary classification task

    # Initialize the LSTM model
    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to PyTorch tensors
    X_training_tensor = torch.tensor(text_feature_training)
    y_training_tensor = torch.tensor(y_training, dtype=torch.int)
    X_validate_tensor = torch.tensor(text_feature_validate)
    y_validate_tensor = torch.tensor(y_validate, dtype=torch.int)

    # Create TensorDataset
    train_dataset = TensorDataset(X_training_tensor, y_training_tensor)
    validate_dataset = TensorDataset(X_validate_tensor, y_validate_tensor)

    # Use DataLoader to load data
    batch_size = 64  # Adjust batch size as needed
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=1)

    # Model training
    num_epochs = 5
    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
    
    # Model evaluation
    model.eval()
    y_text_pred = []

    with torch.no_grad():
        for inputs, labels in validate_dataloader:
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # Predict
            outputs = model(inputs)
            predictions = (outputs >= 0.5).int()
            
            # Save predictions
            y_text_pred.extend(predictions.numpy().tolist())
    
    # Use XGBoost for graph feature
    clf = XGBClassifier(n_estimators=100, max_depth=25, objective='binary:logistic', n_jobs=-1, random_state=0)
    clf.fit(graph_feature_training, y_training)

    y_graph_pred = clf.predict(graph_feature_validate).tolist()
    
    # Combine predictions from LSTM and XGBoost
    combined_predictions = np.column_stack((y_text_pred, y_graph_pred))

    # Initialize logistic regression model (or other model)
    logistic_model = LogisticRegression()

    # Train logistic regression model
    logistic_model.fit(combined_predictions, y_validate)

    # Predict test set
    test_labels = {}
    model.eval()
    with torch.no_grad():
        for transcription_id in test_set:
            with open(path_to_test / f"{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            
            # Get text features for the test set
            X_text_test = []
            for utterance in transcription:
                X_text_test.append(utterance["speaker"] + ": " + utterance["text"])
            
            X_text_test = bert.encode(X_text_test)
            X_text_test = torch.tensor(X_text_test).unsqueeze(1)

            # Predict using the LSTM model
            outputs = model(X_text_test)
            y_text_test = (outputs >= 0.5).int()
            y_text_test = y_text_test.squeeze(1)
            
            # Get graph features for the test set
            with open(path_to_test / f"{transcription_id}.txt", "r") as graph_file:
                edges = []
                relations = []
                for line in graph_file:
                    parts = line.split()
                    source, relation, target = int(parts[0]), parts[1], int(parts[2])
                    
                    edges.append((source, target, {'relation': relation}))
                    relations.append(relation_mapping[relation])
                
            G = nx.DiGraph()
            G.add_edges_from(edges)
            
            node_degrees = dict(G.degree())
            
            # Add centrality measures, using degree centrality as an example
            degree_centrality = nx.degree_centrality(G)
            
            # Handle leaf nodes, set relation to nan
            for node in G.nodes:
                if G.out_degree(node) == 0:
                    relations.append(0)  # Map np.nan to 0
            
            # Combine node degrees, relations, and centrality measures
            X_graph_test = list(zip(node_degrees.values(), relations, degree_centrality.values()))
            
            # Predict using XGBoost
            y_graph_test = clf.predict(X_graph_test)
            
            # Combine predictions from LSTM and XGBoost
            combined_predictions = np.column_stack((y_text_test.numpy(), y_graph_test))
            
            # Predict using logistic regression
            y_test = logistic_model.predict(combined_predictions)
            
            test_labels[transcription_id] = y_test.tolist()
    
    return test_labels


def process_GNN(model_name, training_set, validate_set, test_set, path_to_training, path_to_validate, path_to_test, path_to_labels):
    # Split the dataset into training, validation, and test sets
    training_set, validate_set, test_set = split_dataset(validate=True)

    # Get combined features and relation mapping for training and validation sets
    train_dataset, relation_mapping = get_combine_feature(training_set, path_to_training, path_to_labels)
    validate_dataset, _ = get_combine_feature(validate_set, path_to_validate, path_to_labels, relation_mapping)

    # Create DataLoader for batch processing
    train_loader = GNNDataLoader(train_dataset, batch_size=1, shuffle=True)
    validate_loader = GNNDataLoader(validate_dataset, batch_size=1, shuffle=True)
    
    # Initialize GNN model, loss function, and optimizer based on the specified model_name
    if model_name == "GCN":
        model = GCNModel(input_dim=384, hidden_dim=64, output_dim=2)
    elif model_name == "GAT":
        model = GATModel(input_dim=384, hidden_dim=128, output_dim=2, num_heads=2)
    elif model_name == "GraphSAGE":
        model = GraphSAGEModel(input_dim=384, hidden_dim=32, output_dim=2, dropout=0.2)
    else:
        raise ValueError("model_name must be one of 'GCN', 'GAT' and 'GraphSAGE'.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Train the GNN model
    best_val_f1 = 0.5
    num_epochs = 90
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            y = batch.y
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate the model on the validation set
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in validate_loader:
                output = model(batch)
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        # Calculate F1-Score
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        # Save the model if it achieves the best F1-Score on the validation set
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
        if (epoch+1) % 10 == 0:
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, F1-Score: {f1}")
    
    # Load the best model based on validation F1-Score
    model.load_state_dict(torch.load("best_model.pth"))
    print("Best validation F1-Score: ", best_val_f1)
    
    # Predict labels for the test set using the trained model
    test_labels = {}
    model.eval()
    with torch.no_grad():
        for transcription_id in test_set:       
            with open(path_to_test / f"{transcription_id}.txt", "r") as graph_file:
                lines = graph_file.readlines()

            edges_list = []
            for line in lines:
                parts = line.split()
                if len(parts) == 3:
                    src, relation, dest = int(parts[0]), parts[1], int(parts[2])
                    edges_list.append((src, dest, relation_mapping[relation]))

            # Read node attributes from the JSON file
            text_feature = []
            with open(path_to_test / f"{transcription_id}.json", "r") as text_file:
                transcription = json.load(text_file)
            for utterance in transcription:
                text_feature.append(utterance["speaker"] + ": " + utterance["text"])
            node_features = bert.encode(text_feature)
            node_attributes = torch.tensor(node_features)

            # Create PyTorch Geometric Data object
            x = torch.tensor(node_attributes, dtype=torch.float)

            # Convert the edge list to PyTorch Geometric edge_index
            src_nodes, dest_nodes, relations = zip(*edges_list)
            edge_index = torch.tensor([src_nodes, dest_nodes], dtype=torch.long)

            # Convert edge attributes to PyTorch Geometric edge_attr
            edge_attr = torch.tensor(relations, dtype=torch.float).view(1, -1)

            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1).int()
            test_labels[transcription_id] = predictions.numpy().tolist()
    
    return test_labels