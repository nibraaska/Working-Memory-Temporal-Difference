class agent:
    # Gets the left and right spots
    def get_moves(self, state, size_of_maze):
        if(state == 0):
            return size_of_maze - 1, 1
        elif(state == self.size - 1):
            return size_of_maze, 0
        else:
            return size_of_maze - 1, state + 1