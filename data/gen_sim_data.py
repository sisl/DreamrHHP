from IPython import embed
import json

# Generate positions and routes for each car and each episode
car1_ep1 = {'pos': (42.33,76.85), 'route': {'1' : ((42.45,76.54),24.57), 
            '2' : ((42.67,76.32), 61.45), '3' : ((42.21,76.02), 83.42)}} 
car2_ep1 = {'pos': (40.98,72.56), 'route': {'1' : ((40.80,72.77),9.43)}}

# TODO - The epoch duration is 5 seconds, but the updated time for the 
# car at each waypoint is not necessarily 5 seconds less than the previous epoch!
car1_ep2 = {'pos': (42.36,76.77), 'route': {'1' : ((42.45,76.54),21.88), 
            '2' : ((42.67,76.32), 57.76), '3' : ((42.21,76.02), 77.11)}}
car2_ep2 = {'pos': (40.91,72.69), 'route': {'1' : ((40.80,72.77),3.22)}}

car1_ep3 = {'pos': (42.45,76.64), 'route': {'1' : ((42.45,76.54),16.88), 
            '2' : ((42.67,76.32), 51.95), '3' : ((42.21,76.02), 71.20)}}
car2_ep3 = {'pos': (40.80,72.77), 'route': None}

# Concatenate cars for each epoch
epoch1 = {'1' : car1_ep1, '2' : car2_ep1}
epoch2 = {'1' : car1_ep2, '2' : car2_ep2}
epoch3 = {'1' : car1_ep3, '2' : car2_ep3}

# Setup an episode with three epochs.
# The time difference between the epochs is 5 seconds
episode = {'0' : epoch1, '1' : epoch2, '2' : epoch3}

with open('example_1ep.json','w') as fp:
    json.dump(episode,fp,indent=2)