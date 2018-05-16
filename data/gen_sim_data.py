from IPython import embed
import json

# Generate positions and routes for each car and each episode
car1_ep1 = {'pos': (42.33,76.85), 'route': {'1' : ((42.45,76.54),24.57), 
            '2' : ((42.67,76.32), 61.45), '3' : ((42.21,76.02), 83.42)}} 
car2_ep1 = {'pos': (40.98,72.56), 'route': {'1' : ((40.80,72.77),9.43)}}

# The ETA at various points is changing from epoch to epoch
car1_ep2 = {'pos': (42.36,76.77), 'route': {'1' : ((42.45,76.54),21.88), 
            '2' : ((42.67,76.32), 63.76), '3' : ((42.21,76.02), 83.42)}}
car2_ep2 = {'pos': (40.91,72.69), 'route': {'1' : ((40.80,72.77),8.22)}}

car1_ep3 = {'pos': (42.45,76.64), 'route': {'1' : ((42.45,76.54),21.88), 
            '2' : ((42.67,76.32), 61.95), '3' : ((42.21,76.02), 81.20)}}
car2_ep3 = {'pos': (40.80,72.77), 'route': None}

# Concatenate cars for each epoch
epoch1 = {'time': 0.0, 'car-info': {'car-1' : car1_ep1, 'car-2' : car2_ep1}}
epoch2 = {'time': 5.0, 'car-info': {'car-1' : car1_ep2, 'car-2' : car2_ep2}}
epoch3 = {'time': 10.0,'car-info': {'car-1' : car1_ep3, 'car-2' : car2_ep3}}

# Setup an episode with three epochs.
# The time difference between the epochs is 5 seconds
episode = {'epoch-1' : epoch1, 'epoch-2' : epoch2, 'epoch-3' : epoch3, 'num_epochs' : 3, 'start_pos' : (34.25,81.34), 'goal_pos': (42.56,91.99)}

with open('example_1ep.json','w') as fp:
    json.dump(episode,fp,indent=2)