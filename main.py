from base_model import MatchMakingModel
import data_comparison

# For 2nd question

# Define the file name and Instantiating the MatchMakingModel
file_name= 'MCDA5511_classmates_Sheet1.csv'
model=MatchMakingModel()
attendees_map= model.csv_to_dict(file_name)

# Define a selected list of classmates for data comparison and modifying their description and storing in dictionary
selected_list = [
    "Rakshit Gupta",
    "Neeyati Mehta",
    "Sylvester Terdoo",
    "Tejasvi Bhutiyal",
    ]

classmates_map_data_change = {}
classmates_map_data_change["Rakshit Gupta"] = "I love to travel and uncovering unfamiliar places, and relish engaging in pastimes such as reading and basketball."
classmates_map_data_change["Neeyati Mehta"] = "I like napping."
classmates_map_data_change["Sylvester Terdoo"] = "i enjoy spending time inside"
classmates_map_data_change["Tejasvi Bhutiyal"] = "I love to watch netflix series and discover new places."

# Create a dictionary to store earlier data for selected classmates
classmates_map_earlier_data = {}
for name in selected_list:
    classmates_map_earlier_data[name] = attendees_map[name]

# Compare the modified data using cosine and Euclidean distance metrics
cosine_similarities= data_comparison.compare_data(classmates_map_earlier_data, classmates_map_data_change)
eucledian_similarities= data_comparison.compare_data(classmates_map_earlier_data, classmates_map_data_change, 'euclidean')

# Print comparison result
print('Cosine similarities between old and modifies data:',cosine_similarities)
print('Eucledian similarities between old and modifies data:',eucledian_similarities)
