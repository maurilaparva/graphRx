# __init__.py for Drug Repurposing Discovery with Graph Neural Nets

import os
import sys

# Add the path of the src folder to the system path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loading import load_drkg, preprocess_data
from graph_construction import construct_graph
from model import GDRnet, SIGNEncoder, QuadraticDecoder
from training import train_model, evaluate_model
from utils import visualize_neighborhood, get_n_hop_neighbors, print_neighborhood_stats

def main():
    """
    Main function to run the full drug repurposing prediction pipeline.
    """
    print("Loading and preprocessing data...")
    triplets_df, entities_df, relations_df = load_drkg()
    preprocessed_data = preprocess_data(triplets_df, entities_df, relations_df)
    print("Constructing heterogeneous graph...")
    hg = construct_graph(preprocessed_data)
    print("Defining the model...")
    model = GDRnet(in_feats=128, hidden_feats=64, out_feats=32, num_hops=2)
    print("Training the model...")
    train_model(hg, model)
    print("Evaluating the model...")
    accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate_model(hg, model)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print("Visualizing neighborhood of a random node...")
    neighborhood = get_n_hop_neighbors(hg, n_hops=2, seed=42)
    visualize_neighborhood(neighborhood)
    print_neighborhood_stats(hg, neighborhood)

if __name__ == "__main__":
    main()
