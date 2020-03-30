class EarlyStopping():

    def __init__(self, patience=5):
        self.patience = patience
        self.cnt = 0
        self.loss = []
        self.best_loss = None

    def __call__(self, eval_loss): # one number, not an array
        if self.best_loss is None:
            self.best_loss = eval_loss
        elif eval_loss < self.best_loss:
            self.cnt = 0
            self.best_loss = eval_loss
        else:
            self.cnt += 1
            if self.cnt >= self.patience: # early stopping
                return True
        self.loss.append(eval_loss)
        return False