class agent:
    # Gets the left and right spots
    def get_moves(self, state, size_of_maze):
        if(state == 0):
            return size_of_maze - 1, 1
        elif(state == size_of_maze - 1):
            return size_of_maze - 2, 0
        else:
            return state - 1, state + 1
        
    def pick(self, left, right, atr, wm, nn):
        pass