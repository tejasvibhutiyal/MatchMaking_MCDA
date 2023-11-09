from base_model import MatchMakingModel
import optuna
import model_improvement
from umap_enhancement import load_embeddings
import config as CONFIG
import numpy as np
from sklearn.manifold import TSNE

# Create an Optuna study for optimizing TSNE hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(model_improvement.tsne_objective, n_trials=100)

# Retrieve the best hyperparameters from the study
best_params = study.best_params
print("Best Parameters:", best_params)

path = "MCDA5511_classmates_Sheet1.csv"
embeddings_dict = load_embeddings(path)
embeddings = np.array(list(embeddings_dict.values()))

# Instantiate the TSNE model with the best hyperparameters and set the random state
best_tsne = TSNE(**best_params, random_state=CONFIG.RANDOM_STATE)
optimized_embedding_2d = best_tsne.fit_transform(embeddings)
model = MatchMakingModel()

# Plot and annotate the optimized TSNE visualization
model.plot_and_annotate(
    optimized_embedding_2d,
    embeddings_dict,
    f"visualization_tsne_optimised_{CONFIG.RANDOM_STATE}"
)