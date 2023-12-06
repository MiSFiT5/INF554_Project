import json
from collections import Counter
from sentence_transformers import SentenceTransformer
import networkx as nx
import torch
from torch_geometric.data import Data

# Load pre-trained SentenceTransformer model
bert = SentenceTransformer('all-MiniLM-L6-v2')

def flatten(list_of_list):
    """Flatten a list of lists."""
    return [item for sublist in list_of_list for item in sublist]

def split_dataset(validate=False):
    """Split the dataset into training, validation, and test sets."""
    # List of training transcripts
    training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
    training_set = flatten([[m_id + s_id for s_id in 'abcd'] for m_id in training_set])
    training_set.remove('IS1002a')
    training_set.remove('IS1005d')
    training_set.remove('TS3012c')

    # List of test transcripts
    test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
    test_set = flatten([[m_id + s_id for s_id in 'abcd'] for m_id in test_set])

    if validate:
        # Randomly select 15% of the training set as the validation set
        import random
        random.seed(6969)
        validate_set = random.choices(training_set, k=int(len(training_set) * 0.15))
        training_set = list(set(training_set) - set(validate_set))
        return training_set, validate_set, test_set

    return training_set, test_set

def get_text_feature(dataset, path, show_progress_bar=True):
    """Extract text features using SentenceTransformer."""
    text_feature = []
    for transcription_id in dataset:
        with open(path / f"{transcription_id}.json", "r") as text_file:
            transcription = json.load(text_file)

        for utterance in transcription:
            text_feature.append(utterance["speaker"] + ": " + utterance["text"])

    # Encode text features using SentenceTransformer
    text_feature = bert.encode(text_feature, show_progress_bar=show_progress_bar)
    return text_feature

def get_graph_feature(dataset, path, relation_mapping=None):
    """Extract graph features from the transcripts."""
    graph_feature = []

    if relation_mapping is None:
        # Get relation frequency
        relation_frequency = Counter()

        for transcription_id in dataset:
            with open(path / f"{transcription_id}.txt", "r") as graph_file:
                for line in graph_file:
                    parts = line.split()
                    relation = parts[1]
                    relation_frequency[relation] += 1

        # Sort relations by frequency in descending order
        sorted_relations = sorted(relation_frequency.keys(), key=lambda x: relation_frequency[x], reverse=True)

        relation_mapping = {'nan': 0}  # Map np.nan to 0
        next_relation_id = 1
        for relation in sorted_relations:
            relation_mapping[relation] = next_relation_id
            next_relation_id += 1

    for transcription_id in dataset:
        with open(path / f"{transcription_id}.txt", "r") as graph_file:
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

        # Combine node degrees, relations, and centrality measures as graph feature
        combined_feature = list(zip(node_degrees.values(), relations, degree_centrality.values()))
        graph_feature.extend(combined_feature)

    return graph_feature, relation_mapping

def get_combine_feature(dataset, path, label_file, relation_mapping=None):
    """Combine text and graph features for the dataset."""
    graph_dataset = []
    with open(label_file, "r") as file:
        all_labels = json.load(file)

    if relation_mapping is None:
        # Get relation frequency
        relation_frequency = Counter()

        for transcription_id in dataset:
            with open(path / f"{transcription_id}.txt", "r") as graph_file:
                for line in graph_file:
                    parts = line.split()
                    relation = parts[1]
                    relation_frequency[relation] += 1

        # Sort relations by frequency in descending order
        sorted_relations = sorted(relation_frequency.keys(), key=lambda x: relation_frequency[x], reverse=True)

        relation_mapping = {'nan': 0}  # Map np.nan to 0
        next_relation_id = 1
        for relation in sorted_relations:
            relation_mapping[relation] = next_relation_id
            next_relation_id += 1
            
    for transcription_id in dataset:
        # Read graph data
        with open(path / f"{transcription_id}.txt", "r") as graph_file:
            lines = graph_file.readlines()

        edges_list = []
        for line in lines:
            parts = line.split()
            if len(parts) == 3:
                src, relation, dest = int(parts[0]), parts[1], int(parts[2])
                edges_list.append((src, dest, relation_mapping[relation]))

        # Read node attributes
        text_feature = []
        with open(path / f"{transcription_id}.json", "r") as text_file:
            transcription = json.load(text_file)
        for utterance in transcription:
            text_feature.append(utterance["speaker"] + ": " + utterance["text"])
        node_features = bert.encode(text_feature)
        node_attributes = torch.tensor(node_features)

        # Create PyTorch Geometric Data object
        x = torch.tensor(node_attributes, dtype=torch.float)

        # Convert edge list to PyTorch Geometric edge_index
        src_nodes, dest_nodes, relations = zip(*edges_list)
        edge_index = torch.tensor([src_nodes, dest_nodes], dtype=torch.long)

        # Convert edge attributes to PyTorch Geometric edge_attr
        edge_attr = torch.tensor(relations, dtype=torch.float).view(1, -1)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(all_labels[transcription_id]))

        graph_dataset.append(data)

    return graph_dataset, relation_mapping

def get_label(dataset, label_file):
    """Get labels for the dataset."""
    labels = []
    with open(label_file, "r") as file:
        all_labels = json.load(file)
    for transcription_id in dataset:
        labels += all_labels[transcription_id]
    return labels
