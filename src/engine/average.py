import sklearn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccuracyMeter:
    def __init__(self):        
        self.reset()
     
        
    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_true, y_pred, batch_size=1):

        # so we just need to count total num of images / batch_size
        #self.count += num_steps
        self.batch_size = batch_size
        self.count += self.batch_size
        # this part here already got an acc score for the 4 images, so no need divide batch size
        self.score = sklearn.metrics.accuracy_score(y_true, y_pred)
        total_score = self.score * self.batch_size

        self.sum += total_score
        

    @property
    def avg(self):        
        self.avg_score = self.sum/self.count
        return self.avg_score