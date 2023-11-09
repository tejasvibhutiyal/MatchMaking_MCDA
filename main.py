from base_model import MatchMakingModel
import model_comparison


"""
Main execution block to compute and print the Spearman's rank correlation for a specific person's embeddings
generated from two different transformer models.

Note:
    This block reads classmate data, generates embeddings using two different transformer models,
    and computes the Spearman's rank correlation for a specific individual's embeddings from both models.
"""
file_name = 'MCDA5511_classmates_Sheet1.csv' 
model= MatchMakingModel()
classmates_map = model.csv_to_dict(file_name)
person_embeddings_minilm_l6_v2 = model.generate_embeddings(classmates_map, 'sentence-transformers/all-MiniLM-L6-v2')

person_embeddings_mpnet_base_v2 = model.generate_embeddings(
    classmates_map, 'sentence-transformers/all-mpnet-base-v2'
)
print("created embeddings using person_embeddings_mpnet_base_v2")

correlation = model_comparison.rank_correlation(
    person_embeddings_minilm_l6_v2, person_embeddings_mpnet_base_v2, "Rakshit Gupta"
)
print(f"Spearman's rank correlation: {round(correlation * 100,3)}%")