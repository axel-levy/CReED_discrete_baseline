import numpy as np

class EventsArray():
    def __init__(self, events_path):
        self.event_block = np.load(events_path)
        
        self.n_events = self.event_block.shape[0]

        t_beg = self.event_block[0, 3]
        t_end = self.event_block[-1, 3]
        
        self.t_beg = t_beg
        self.t_end = t_end

        self.event_block[:, 3] = (self.event_block[:, 3] - t_beg) / (t_end - t_beg)

        self.event_block[:,3] = 2. * self.event_block[:, 3] - 1.
