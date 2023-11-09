from base_model import MatchMakingModel
import data_comparison

# for 2nd question

file_name= 'MCDA5511_classmates_Sheet1.csv'
model=MatchMakingModel()
attendees_map= model.csv_to_dict(file_name)

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

classmates_map_earlier_data = {}
for name in selected_list:
    classmates_map_earlier_data[name] = attendees_map[name]

print(classmates_map_earlier_data)
cosine_similarities= data_comparison.compare_data(classmates_map_earlier_data, classmates_map_data_change)
eucledian_similarities= data_comparison.compare_data(classmates_map_earlier_data, classmates_map_data_change, 'euclidean')
print('Cosine similarities between old and modifies data:',cosine_similarities)
print('Eucledian similarities between old and modifies data:',eucledian_similarities)
