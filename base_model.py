import csv
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
from collections import defaultdict
import config as CONFIG
import os

class MatchMakingModel:
    """
    A class representing a matchmaking model for pairing attendees based on embeddings.

    """

    def csv_to_dict(self,file_name):
        """
        Reads data from a CSV file containing names and description,
        and returns a dictionary mapping names to corresponding description.

        Parameters:
        - file_name (str): The name of the CSV file to be read.

        Returns:
        dict: A dictionary where keys are names and values are paragraphs.
        """
        attendees_map = {}
        with open(file_name, newline='',encoding="utf8") as csvfile:
            attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(attendees)  # Skip the header row
            for row in attendees:
                name, paragraph = row
                attendees_map[name] = paragraph
        
        return attendees_map

    def generate_embeddings(self,attendees_map):
        """
        Generates sentence embeddings for description in the provided attendees map using a pre-trained model.

        Parameters:
        - attendees_map (dict): A dictionary where keys are names and values are corresponding description.

        Returns:
        list: A list of sentence embeddings generated from the descriptions.
        """
        names = list(attendees_map.keys())
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        descriptions = list(attendees_map.values())
        embeddings= model.encode(descriptions)
        person_embeddings = {name: embedding for name, embedding in zip(names, embeddings)}
        return person_embeddings
    
    def reduce_dimensionality(self, person_embeddings):
        """
        Reduces the dimensionality of person embeddings using UMAP (Uniform Manifold Approximation and Projection).

        Parameters:
        - person_embeddings (dict): A dictionary where keys are names and values are corresponding embeddings.

        Returns:
        numpy.ndarray: An array containing the reduced-dimensional representations of person embeddings.
        """
        reducer = umap.UMAP(random_state= CONFIG.RANDOM_STATE)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(list(person_embeddings.values()))
        reduced_data = reducer.fit_transform(scaled_data)
        return reduced_data

    def plot_and_annotate(self, reduced_data, person_embeddings, file_name='visualization.png', dpi=800):
        """
        Plots and annotates data points in a 2D scatter plot based on reduced-dimensional embeddings.

        Parameters:
        - reduced_data (numpy.ndarray): An array containing the reduced-dimensional representations of person embeddings.
        - person_embeddings (dict): A dictionary where keys are names and values are corresponding embeddings.
        - file_name (str, optional): The file path to save the visualization. Default is 'visualization.png'.
        - dpi (int, optional): Dots per inch for the saved image. Default is 800.
        """
        # Extracting coordinates and labels
        x = [row[0] for row in reduced_data]
        y = [row[1] for row in reduced_data]
        labels = list(person_embeddings.keys())

        # Plotting and annotating data points
        plt.scatter(x, y)
        for i, name in enumerate(labels):
            plt.annotate(name, (x[i], y[i]), fontsize=3)

        # Clean-up and Export
        file_name = os.path.join(CONFIG.BASE_RESULTS,file_name)
        plt.axis('off')
        plt.savefig(file_name, dpi=dpi)
        plt.show()

    def top_matches(self, attendees_map, person_embeddings):
        """
        Computes and returns the top matching attendees for each person based on cosine similarity of embeddings.

        Parameters:
        - attendees_map (dict): A dictionary where keys are names and values are corresponding paragraphs.
        - person_embeddings (dict): A dictionary where keys are names and values are corresponding embeddings.

        Returns:
        dict: A dictionary where keys are names, and values are lists of top matching attendees and their cosine similarities.
        """
        top_matches = {}
        all_personal_pairs = defaultdict(list)
        for person in attendees_map.keys():
            for person1 in attendees_map.keys():
                all_personal_pairs[person].append([spatial.distance.cosine(person_embeddings[person1], person_embeddings[person]), person1])

        for person in attendees_map.keys():
            top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[1])
        
        return top_matches

