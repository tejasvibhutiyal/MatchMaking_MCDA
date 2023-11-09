from scipy.stats import spearmanr
from scipy.spatial.distance import cosine


def compute_similarity(embeddings_dict, reference_embedding):
    """
    Computes the cosine similarity between a reference embedding and all other embeddings in a dictionary.

    Args:
        embeddings_dict (dict): A dictionary of embeddings with keys as names and values as embedding vectors.
        reference_embedding (list): The embedding vector of the reference individual.

    Returns:
        dict: A dictionary of similarities with keys as names and values as the similarity score with the reference embedding.
    """
    similarities = {}
    for name, emb in embeddings_dict.items():
        similarity = 1 - cosine(reference_embedding, emb)
        similarities[name] = similarity
    return similarities


def rank_correlation(model1_embeddings, model2_embeddings, person_ref_embedding):
    """
    Computes the Spearman's rank correlation between the similarity rankings of two different embedding models for a reference person.

    Args:
        model1_embeddings (dict): Embeddings dictionary from the first model.
        model2_embeddings (dict): Embeddings dictionary from the second model.
        person_ref_embedding (str): The name of the reference person whose embedding is to be compared.

    Returns:
        float: The Spearman's rank correlation coefficient between the two sets of rankings.
    """
    similarities_model1 = compute_similarity(
        model1_embeddings, model1_embeddings[person_ref_embedding]
    )
    similarities_model2 = compute_similarity(
        model2_embeddings, model2_embeddings[person_ref_embedding]
    )

    # Sorting the names based on similarity
    sorted_names_model1 = sorted(
        similarities_model1, key=similarities_model1.get, reverse=True
    )
    sorted_names_model2 = sorted(
        similarities_model2, key=similarities_model2.get, reverse=True
    )

    # Creating rank lists
    ranks_model1 = [sorted_names_model1.index(name) for name in similarities_model1]
    ranks_model2 = [sorted_names_model2.index(name) for name in similarities_model1]

    # Compute Spearman's rank correlation
    correlation, _ = spearmanr(ranks_model1, ranks_model2)
    return correlation