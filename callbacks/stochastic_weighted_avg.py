from tensorflow.keras.callbacks import Callback

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
                weights_total = None
                for j in range(self.num_weights):
                    if j == 0:
                        weights_total = self.swa_weights_list[j][i]
                    else:
                        weights_total += self.swa_weights_list[j][i]
                self.swa_weights[i] = weights_total/self.num_weights 
        else:
            self.swa_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.params['epochs'] >= self.num_weights:
            self.model.set_weights(self.swa_weights)
            print(f'Final model parameters set to stochastic weight average of final {self.num_weights} epochs.')
            self.model.save(self.filepath)
            print('Final stochastic averaged weights saved to file.')