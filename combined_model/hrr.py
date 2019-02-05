import numpy as np

class holographic_reduced_representations:
    
    # Lenght of hrrs
    def __init__(self, length):
        self.length = length
        
    # Creating hrrs
    def hrr(self, normalized=False):
        if normalized:
            x = np.random.uniform(-np.pi,np.pi,int((self.length-1)/2))
            if self.length % 2:
                x = np.real(np.fft.ifft(np.concatenate([np.ones(1), np.exp(1j*x), np.exp(-1j*x[::-1])])))
            else:
                x = np.real(np.fft.ifft(np.concatenate([np.ones(1), np.exp(1j*x), np.ones(1), np.exp(-1j*x[::-1])])))
        else:
            x = np.random.normal(0.0, 1.0/np.sqrt(self.length), self.length)
        return x
    
    # Creates an array of hrrs of size size_of_array x size_of_hrrs
    def hrr_array(self, size_of_array, normalized=False):
        hrr_arr = np.zeros([size_of_array, self.length])
        for x in range(size_of_array):
            hrr_arr[x] = self.hrr(normalized)
        return hrr_arr
    
    # Convolve two hrrs
    def convolve(self, x, y):
        return np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))

    # Preconvolve all hrrs to save time
    def preconvolve(self, possible_wm, possible_signals, state_hrrs):
        preconvolved_matrix = np.zeros([possible_wm.size, possible_signals.size, state_hrrs.size, self.length])
        for x in range(len(possible_wm)):
            for y in range(len(possible_signals)):
                for z in range(len(state_hrrs)):
                    preconvolved_matrix[x][y][z] = self.convolve(self.convolve(possible_wm[x], possible_signals[y]), state_hrrs[z])
        return preconvolved_matrix