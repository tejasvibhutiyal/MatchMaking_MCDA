from base_model import MatchMakingModel


file_name= 'MCDA5511_classmates_Sheet1.csv'
model=MatchMakingModel()
attendees_map= model.csv_to_dict(file_name)
person_embeddings= model.generate_embeddings(attendees_map)
reduced_data= model.reduce_dimensionality(person_embeddings)    
model.plot_and_annotate(reduced_data, person_embeddings)
topmatches= model.top_matches(attendees_map, person_embeddings)
print(topmatches)