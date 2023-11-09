import umap
import numpy as np
import json
import optuna
import os
from base_model import MatchMakingModel
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import config as CONFIG


def load_embeddings(file_path):
    """
    Loads embeddings from a JSON file.

    Args:
        file_path (str): The file path of the data file.

    Returns:
        dict: A dictionary containing the loaded embeddings.
    """
    model = MatchMakingModel()
    attendees_map = model.csv_to_dict(file_path)
    embeddings = model.generate_embeddings(attendees_map)

    return embeddings


def umap_objective(trial):
    """
    Objective function for optimizing UMAP hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): A trial instance from Optuna.

    Returns:
        float: The mean Spearman correlation coefficient across all pairs of embeddings.

    Note:
        This function suggests hyperparameters (n_neighbors, min_dist, metric) for UMAP,
        loads embeddings, applies dimensionality reduction, and calculates the mean Spearman
        correlation coefficient between cosine similarity ranks and Euclidean distance ranks of the embeddings.
    """
    n_neighbors = trial.suggest_int("n_neighbors", 2, 50)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.9)
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=CONFIG.RANDOM_STATE
    )

    path = "MCDA5511_classmates_Sheet1.csv"
    embeddings_dict = load_embeddings(path)
    embeddings = np.array(list(embeddings_dict.values()))
    embedding_2d = reducer.fit_transform(embeddings)

    correlations = []
    for i in range(len(embeddings)):
        cos_similarities = np.array(
            [1 - cosine(embeddings[i], emb) for emb in embeddings]
        )

        euclidean_dists = np.linalg.norm(embedding_2d - embedding_2d[i], axis=1)

        cos_ranks = np.argsort(-cos_similarities)
        euclidean_ranks = np.argsort(euclidean_dists)

        correlation, _ = spearmanr(cos_ranks, euclidean_ranks)
        correlations.append(correlation)

    return np.mean(correlations)
