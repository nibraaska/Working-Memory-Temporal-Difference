import random

# Goals must be a 2d array of integers: num_of_non_obs_tasks x num_of_obs_tasks. 
# All possible goals must be put into the array for all possible combination of non observable task switching and obervable task switching.
class maze:
    # Size of the maze
    def __init__(self, size_of_maze, non_obs_task_switch_rate, num_of_non_obs_tasks, num_of_obs_tasks, goals):
        self.size = size_of_maze
        
        # Non observable task switching variables
        self.non_obs_task_switch = non_obs_task_switch_rate
        self.num_of_non_obs_tasks = num_of_non_obs_tasks
        self.non_obs_count = 0
        self.switch_non_obs = 0
        
        # Observable task switching variables
        self.num_of_obs_tasks = num_of_obs_tasks
        self.obs_count = 0
        self.switch_obs = 0
        
        # Goals for the maze with respect to task switches
        self.goals = goals
        
        self.signal = 0
        
    def step_maze(self, index=None):
        return self.non_obs_switch_func(index)
     
    # Non observable task switching for any number of switches
    def non_obs_switch_func(self, index=None):
        if self.non_obs_count == self.non_obs_task_switch:
            self.non_obs_count = 0
            self.switch_non_obs =  self.switch_non_obs + 1
            if self.switch_non_obs >= self.num_of_non_obs_tasks:
                self.switch_non_obs = 0
        self.non_obs_count += 1
        return self.obs_task_switch_func(self.switch_non_obs, index)
    
    # Observable task switching for any number of switching. Done randomly
    def obs_task_switch_func(self, switch_non_obs, index=None):
        if not index:
            self.signal = random.randint(0, self.num_of_obs_tasks - 1)
        else:
            self.signal = index + 1
        return self.goals[switch_non_obs][self.signal - 1]