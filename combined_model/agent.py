class agent:
    # Gets the left and right spots
    def get_moves(self, state):
        if(state == 0):
            return self.size - 1, 1
        elif(state == self.size - 1):
            return state - 1, 0
        else:
            return state - 1, state + 1