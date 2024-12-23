from numpy import mean
import wandb

class BaseModelLogger():
    def __init__(self, tokenizer, config):
        self.config = config
        self.queries = config.queries

        self.tokenzier = tokenizer
        
        self.performance_dict = {}
        for query in self.queries:
            self.performance_dict[query] = []

    def log(self, output, target, query):
        prediction = output['logits'].argmax(dim = 2).squeeze()[-1].item()
        try:
            target = target['input_ids'].squeeze()[0].item()
        except:
            target = target['input_ids'].item()

        if prediction == target:
            self.performance_dict[query].append(1)
        else:
            self.performance_dict[query].append(0)

    def report(self):
        for query in self.performance_dict.keys():
            history = self.performance_dict[query]
            performance = sum(history)/(len(history) + 1)
            print(str(query) + "\t\t" + str(performance))
        
class ViTPretrainingLogger():
    def __init__(self, config):
        self.config = config
        self.queries = config.queries

        self.performance_dict = {}
        for query in self.queries:
            self.performance_dict[query] = []

    def log(self, loss, query):
        self.performance_dict[query].append(loss)
        wandb.log({f"{query} Loss": loss})


    def report(self):
        for query in self.performance_dict.keys():
            print(str(query) + " Loss:\t\t" + str(mean(self.performance_dict[query])))
            wandb.log({f"{query} Mean Loss": mean(self.performance_dict[query])})
