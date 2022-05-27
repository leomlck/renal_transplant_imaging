from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

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

def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def roc_auc(y_true, y_pred):
        try:
                return roc_auc_score(y_true, y_pred)
        except ValueError:
                return 0


