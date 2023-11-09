from base_model import MatchMakingModel
from scipy.spatial import distance

def compare_data(old_attendees_map, modified_attendees_map, distance_metric = 'cosine'):
    """
    Compare embeddings between old and modified attendee data using a specified distance metric.

    Parameters:
    - old_attendees_map (dict): Dictionary containing names as keys and paragraphs as values for the old data.
    - modified_attendees_map (dict): Dictionary containing names as keys and paragraphs as values for the modified data.
    - distance_metric (str, optional): The distance metric to use for comparison. Default is 'cosine'.
                                     Possible values: 'cosine', 'euclidean', and other valid distance metrics.

    Returns:
    - comparison_results (dict): Dictionary containing names as keys and similarity scores as values.
    """
    model = MatchMakingModel()
    old_attendees_embeddings=model.generate_embeddings(old_attendees_map)
    modified_attendees_embeddings = model.generate_embeddings(modified_attendees_map)
    # Compare embeddings
    comparison_results = {}
    for name in old_attendees_map:
        old_embedding = old_attendees_embeddings[name]
        modified_embedding = modified_attendees_embeddings[name]

        # Calculate distance based on the chosen metric
        if distance_metric == 'cosine':
            similarity = 1 - distance.cosine(old_embedding, modified_embedding)
        elif distance_metric == 'euclidean':
            similarity = -distance.euclidean(old_embedding, modified_embedding)
            # Using negative Euclidean distance as a similarity measure

        # Store the result
        comparison_results[name] = similarity

    return comparison_results





