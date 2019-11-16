from tensorflow.keras.callbacks import Callback
import functools

class SWA(Callback):
    
    def __init__(self, filepath, num_weights):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.num_weights = num_weights
        self.swa_weights_list = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.swa_weights_list.append(self.model.get_weights())
        if len(self.swa_weights_list) > self.num_weights:
            self.swa_weights_list.pop()
            
        if epoch >= self.num_weights:    
            for i in range(len(self.swa_weights)):
                weights = [self.swa_weights_list[j][i] for j in \
                           range(self.num_weights)]
                weights_avg = functools.reduce(lambda a,b : a+b, weights)
                self.swa_weights[i] = weights_avg/self.num_weights 
        else:
            self.swa_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.params['epochs'] >= self.num_weights:
            self.model.set_weights(self.swa_weights)
            print(f'Final model parameters set to stochastic weight average of final {self.num_weights} epochs.')
            self.model.save_weights(self.filepath)
            print('Final stochastic averaged weights saved to file.')