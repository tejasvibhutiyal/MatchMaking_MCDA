from base_model import MatchMakingModel
import config as CONFIG
import umap_enhancement
import optuna
import numpy as np
import umap

file_name= 'MCDA5511_classmates_Sheet1.csv'
model=MatchMakingModel()
attendees_map= model.csv_to_dict(file_name)
randomState=[27,12,42]

# Loop through each random state for the initial UMAP visualization
for random in randomState:
    CONFIG.RANDOM_STATE = random
    person_embeddings= model.generate_embeddings(attendees_map)
    reduced_data= model.reduce_dimensionality(person_embeddings)    
    model.plot_and_annotate(
        reduced_data, 
        person_embeddings,
        f"visualization_umap_RandomState_{CONFIG.RANDOM_STATE}"
        )



# Loop through each random state for UMAP optimization using Optuna
for random in randomState:
    CONFIG.RANDOM_STATE = random

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(umap_enhancement.umap_objective, n_trials=100)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Load embeddings, apply the best UMAP reducer, and plot the optimized UMAP visualization
    embeddings_dict = umap_enhancement.load_embeddings(file_name)
    embeddings = np.array(list(embeddings_dict.values()))
    best_reducer = umap.UMAP(**best_params, random_state=CONFIG.RANDOM_STATE)
    optimized_embedding_2d = best_reducer.fit_transform(embeddings)
    model.plot_and_annotate(
        optimized_embedding_2d,
        embeddings_dict,
        f"visualization_umap_optimised_RandomState_{CONFIG.RANDOM_STATE}"
    )
