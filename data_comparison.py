from base_model import MatchMakingModel
from scipy.spatial import distance

def compare_data(old_attendees_map, modified_attendees_map, distance_metric = 'cosine'):
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
            # Note: Using negative Euclidean distance as a similarity measure

        # Store the result
        comparison_results[name] = similarity

    return comparison_results





