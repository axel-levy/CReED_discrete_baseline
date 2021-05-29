import numpy as np
import os

# Global variables
events_folder = 'events/'
filename = 'events'
duration = 25.
polarity_on_1_bit = True
horizontal_mirror = False
vertical_mirror = False
remove_first_line = False
W = 180
H = 240

def txt_to_events(filename,local_events_folder, duration=25.,polarity_on_1_bit = True, horizontal_mirror = True,vertical_mirror = True, remove_first_line = True, W=180, H=240):
    '''Translates a stream of events from .txt to .npy'''
    save_dir = events_folder+local_events_folder
    os.mkdir(save_dir)
    
    if remove_first_line:
        shift = 1
    else:
        shift = 0
    
    print("Opening file "+local_events_folder+"...")
    file = open('data/'+local_events_folder+'/'+filename+'.txt','r')
    L = file.readlines()
    print("... file successfully opened")
    t_beg = float(L[0].split()[0])
    t_current = t_beg
    i = 0
    print("First timestamp: "+str(t_beg))
    events_list = []
    n_lines = len(L)-shift
    print("Creating stream of events...")
    while t_current - t_beg < duration and i < n_lines:
        line = L[i+shift].split()
        t = float(line[0])
        x = int(line[2])
        if horizontal_mirror:
            x = W-1-x
        y = int(line[1])
        if vertical_mirror:
            y = H-1-y
        p = float(line[3])
        t_current = t
        events_list.append([p,x,y,t])
        i += 1
    events_array = np.array(events_list)
    print("... stream of events created")
    print("Number of events: "+str(events_array.shape[0]))
    if polarity_on_1_bit:
        events_array[:,0] = 2*events_array[:,0]-1.
    print("Saving...")
    np.save(save_dir+"/events.npy",events_array)
    print("... successfully saved")
    return(events_array)

# File 1
local_events_folder = 'shapes_6dof'
events_array = txt_to_events(filename,local_events_folder,duration=duration,polarity_on_1_bit=polarity_on_1_bit,
                             horizontal_mirror=horizontal_mirror,vertical_mirror=vertical_mirror,
                             remove_first_line=remove_first_line,W=W,H=H)
print("Last event: "+str(events_array[-1,3]))
print("Total duration: "+str(events_array[-1,3]-events_array[0,3]))

# File 2
local_events_folder = 'office_zigzag'
events_array = txt_to_events(filename,local_events_folder,duration=duration,polarity_on_1_bit=polarity_on_1_bit,
                             horizontal_mirror=horizontal_mirror,vertical_mirror=vertical_mirror,
                             remove_first_line=remove_first_line,W=W,H=H)
print("Last event: "+str(events_array[-1,3]))
print("Total duration: "+str(events_array[-1,3]-events_array[0,3]))


# File 3
local_events_folder = 'slider_depth'
events_array = txt_to_events(filename,local_events_folder,duration=duration,polarity_on_1_bit=polarity_on_1_bit,
                             horizontal_mirror=horizontal_mirror,vertical_mirror=vertical_mirror,
                             remove_first_line=remove_first_line,W=W,H=H)
print("Last event: "+str(events_array[-1,3]))
print("Total duration: "+str(events_array[-1,3]-events_array[0,3]))




