# -*- coding: utf-8 -*-
# DataUtil/DataLoader.py
import torch
from transformers import AutoModelForCausalLM,  AutoTokenizer
from datasets import load_dataset
import random

from random import choices, randint
from DataUtil.LanguageParser import getParser,  getLanguage
from DataUtil.TreeQuery import getQueryString, getQuery
import torch.nn.functional as tnnf

class IterableQueryLoader(torch.utils.data.IterableDataset):
    def __init__(self, hf_dataset, query_name, max_samples, max_length, lang, model):
        super(IterableQueryLoader).__init__()
        self.hf_dataset = hf_dataset
        self.model = model
        self.lang = lang
        self.max_length = max_length


        self.query_name = query_name
        self.max_samples = max_samples
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        if 'gpt2' not in model.lower():
            if query_name == 'noise':
                pass
            else:
                self.query = getQuery(query_name, self.lang, self.tokenizer)
    
    def __iter__(self):
        i = 0

        if self.query_name == 'noise':
            # if i > self.max_samples:
            #     raise StopIteration
            while i < self.max_samples:
                returnable =  self.process(None)
                i+=1
                yield returnable, self.query_name
        else:
            iterator = iter(self.hf_dataset)
            while i < self.max_samples:
                try:
                    file = next(iterator)
                except StopIteration:
                    iterator = iter(self.hf_dataset)
                    file = next(iterator)
                try:
                    returnable = self.process(file)
                    i+=1
                    yield returnable, self.query_name
                except ValueError:
                    continue

    def __len__(self):
        return len(self.hf_dataset)

    def process(self,  sample):
        if "starcoder" in self.model.lower():
            if self.query_name == 'noise':
                return self.gen_noise_starcoder()
            return self.gen_subsample_starcoder(self.query.get_span(sample['content']))
        elif "gpt" in self.model.lower():
            return self.gen_subsample_gpt(self.tokenize(*self.prep_gpt(sample['ioi_sentences'])))
        else:
            raise ValueError
            
    def gen_noise_starcoder(self):
        tokens = list(self.tokenizer.get_vocab().values())
        noise = choices(tokens, k= self.max_length)
        
        noise[0] = 1
        noise[-1] = 2
        noise[randint(1, self.max_length-2)] = 3
        
        noise = torch.tensor(noise)
        
        sample = {}
        sample['input'] = {'input_ids': noise, 'attention_mask': torch.tensor([1 for _ in noise])}
        return sample

    def gen_subsample_starcoder(self, sample):
        ids = sample['input']['input_ids'].flatten()
        mask = sample['input']['attention_mask'].flatten()
        max = ids.size()[0]
        start = torch.tensor([1])
        stop = torch.tensor([2])
        if max + 2 < self.max_length:
            #pad
            #ids = torch.cat((torch.zeros(self.max_length-max), ids)).int()
            #mask = torch.cat((torch.zeros(self.max_length-max), mask)).int()
            pass
        else:
            #truncate
            #get fim_middle
            fim_id = (ids == 3).nonzero().item()
            if fim_id <= self.max_length//2:
                ids = torch.cat((start, ids))
                ids = ids[:self.max_length-2]
                ids = torch.cat((start,  ids, stop)).int()
                mask = torch.ones(ids.size()).int()
            else:
                context_right = ((self.max_length-2)//2)
                context_left = context_right
                ids = ids[fim_id-context_left:fim_id+context_right]
                ids = torch.cat((start, ids, stop)).int()
                mask = torch.ones(ids.size()).int()

        sample['input']['input_ids'] = ids
        sample['input']['attention_mask'] = mask
        return sample
    
    def prep_gpt(self,  content):
        if self.query_name == 'random':
            tokens = self.tokenizer(content)['input_ids']
            begin = random.randint(0, len(tokens)-15)
            selection = tokens[0:begin]
            target = tokens[begin:begin+5]
            content = self.tokenizer.decode(selection)
            target = self.tokenizer.decode(target)
            return content, target
        
        #This is IOI only
        context = ' '.join(content.split()[0:-1])
        target = ' ' + content.split()[-1]

        return context, target
    
    def gen_subsample_gpt(self, sample):
        return sample
    
    
    def tokenize(self, content, label):
        input = self.tokenizer(content, return_tensors = 'pt')
        label = self.tokenizer(label, return_tensors = 'pt')

        return {"input": input, "label": label}


class IterableScenarioLoader(torch.utils.data.IterableDataset):
    def __init__(self,  hf_dataset,  query_name, max_samples, max_length, lang, model, min_length = 16):
        super(IterableScenarioLoader).__init__()
        self.hf_dataset = hf_dataset
        self.model = model
        self.lang = lang
        self.max_length = max_length
        self.min_length = min_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.query_name = query_name
        self.max_samples = max_samples
        
        if query_name != "random" and 'starcoder' in self.model:
            self.query = getLanguage(self.lang).query(getQueryString(self.lang, query_name))
            self.parser = getParser(self.lang)
        

        
    def __iter__(self):
        i = 0
        iterator = iter(self.hf_dataset)
        while i < self.max_samples:
            try:
                file = next(iterator)
            except StopIteration:
                iterator = iter(self.hf_dataset)
                file = next(iterator)
            try:
                returnable = self.process(file)
                i += 1
                yield returnable, self.query_name
            except ValueError:
                continue

                
    def __len__(self):
        return len(self.hf_dataset)
        
    def process(self,  sample):
        if "starcoder" in self.model.lower():
            return self.gen_subsample_starcoder(self.tokenize(*self.prep_starcoder(sample['content'])))
        elif "gpt" in self.model.lower():
            if self.query_name =='ABBA' or self.query_name =='ABAB':
                if self.query_name == sample['template']:
                    return self.gen_subsample_gpt(self.tokenize(*self.prep_gpt(sample['ioi_sentences'])))
            else:
                return self.gen_subsample_gpt(self.tokenize(*self.prep_gpt(sample['content'])))
        else:
            raise ValueError

    
    def prep_gpt(self,  content):
        if self.query_name == 'random':
            tokens = self.tokenizer(content)['input_ids']
            begin = random.randint(0, len(tokens)-15)
            selection = tokens[0:begin]
            target = tokens[begin:begin+5]
            content = self.tokenizer.decode(selection)
            target = self.tokenizer.decode(target)
            return content, target
        
        #This is IOI only
        context = ' '.join(content.split()[0:-1])
        target = ' ' + content.split()[-1]

        return context, target
        
    def gen_subsample_gpt(self, content):
        ids = content['input']['input_ids'].flatten()
        mask = content['input']['attention_mask'].flatten()
        max = ids.size()[0]
        if max < self.max_length and max > self.min_length:
            #pad
            #ids = torch.cat((torch.zeros(self.max_length-max), ids)).int()
            #mask = torch.cat((torch.zeros(self.max_length-max), mask)).int()
            pass
        else:
            #truncate
            ids = ids[-self.max_length:]
            mask = torch.ones(ids.size()).int()
            
        content['input']['input_ids'] = ids
        content['input']['attention_mask'] = mask
        return content 
    
    def prep_starcoder(self, content):
        if self.query_name == 'random':
            tokens = self.tokenizer(content)['input_ids']
            span_begin = random.randint(0, len(tokens)-15)
            span_end = span_begin + random.randint(3, 10)
            prefix = tokens[0:span_begin]
            postfix = tokens[span_end:]
            target = tokens[span_begin:span_end]

            prefix = self.tokenizer.decode(prefix)
            postfix = self.tokenizer.decode(postfix)
            content = prefix + "<fim_suffix>" + postfix

            target = self.tokenizer.decode(target)
            return content, target
        content = bytes(content, "UTF-8")
        tree = self.parser.parse(content)
        captures = self.query.captures(tree.root_node)
    
        try:
            capture = random.sample(captures, 1)[0]
        except ValueError:
            raise ValueError("No matches detected in sample")
        start = capture[0].start_byte
        finish = capture[0].end_byte
    
        target = content[start:finish]
        content = content[:start] + b"<fim_suffix>" + content[finish:]
    
        content = content.decode("UTF-8")
        target = target.decode("UTF-8")
        return content, target

    def gen_subsample_starcoder(self, sample):
        ids = sample['input']['input_ids'].flatten()
        mask = sample['input']['attention_mask'].flatten()
        max = ids.size()[0]
        start = torch.tensor([1])
        stop = torch.tensor([2])
        if max + 2 < self.max_length:
            #pad
            #ids = torch.cat((torch.zeros(self.max_length-max), ids)).int()
            #mask = torch.cat((torch.zeros(self.max_length-max), mask)).int()
            pass
        else:
            #truncate
            #get fim_middle
            fim_id = (ids == 3).nonzero().item()
            if fim_id <= self.max_length//2:
                ids = torch.cat((start, ids))
                ids = ids[:self.max_length-2]
                ids = torch.cat((start,  ids, stop)).int()
                mask = torch.ones(ids.size()).int()
            else:
                context_right = ((self.max_length-2)//2)
                context_left = context_right
                ids = ids[fim_id-context_left:fim_id+context_right]
                ids = torch.cat((start, ids, stop)).int()
                mask = torch.ones(ids.size()).int()
        
        sample['input']['input_ids'] = ids
        sample['input']['attention_mask'] = mask
        return sample
        
    def tokenize(self, content, label):
        input = self.tokenizer(content, return_tensors = 'pt')
        label = self.tokenizer(label, return_tensors = 'pt')

        return {"input": input, "label": label}

class IterableScenarioAggregator(torch.utils.data.IterableDataset):
    def __init__(self, dataset, max_samples, max_length, queries, lang, model, split):
        self.max_samples = max_samples
        self.max_length = max_length
        self.queries = queries
        self.lang = lang
        self.model = model
        # self.hf_dataset = load_dataset(
        #     self.dataset_location,
        #     self.subset_name,
        #     split=split,
        #     num_proc=64,
        #     # optional use specific cache rather than global hugginface cache
        #     # cache_dir="./cache"
        # )
        self.dataset = dataset.shuffle()
        self.reset()
        self.max_count = len(self.queries) * self.max_samples
        
    def __iter__(self):
        while len(self.scenario_iterators) > 0:
            try:
                iterator = random.choice(self.scenario_iterators)
                next_item = next(iterator)
                self.count += 1
                yield next_item
            except StopIteration:
                self.scenario_iterators.remove(iterator)
                if len(self.scenario_iterators) == 0:
                    self.reset()
                    return
            except ValueError:
                continue
                
    def __len__(self):
        """
        Returns the remaining amount of samples available to iterate over.
        """
        return self.max_count - self.count
                
    def reset(self):
        """
        All samples are available again.
        """
        self.count = 0

        self.scenario_iterators = []
        for query in self.queries:
            self.scenario_iterators.append(iter(IterableQueryLoader(self.dataset,  query, self.max_samples, self.max_length, self.lang, self.model)))



class IterableAttentionLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset,  max_samples, max_length, queries, lang, model, correct_only, target_model, target_model_device, split, evaluation = False, min_length = 16):
        self.target_model = target_model
        self.target_model_device = target_model_device
        self.correct_only = correct_only
        self.dataset = dataset
        self.max_samples = max_samples
        self.max_length = max_length
        self.min_length = 16
        self.queries = queries
        self.lang = lang
        self.model_name = model
        self.split = split
        self.max_count = len(self.queries) * self.max_samples
        self.count = 0
        self.scenario_aggregator = IterableScenarioAggregator(self.dataset, 500000, self.max_length, self.queries, self.lang, self.model_name, self.split)
        self.evaluation = evaluation


        self.reset()


    def __iter__(self):
        try:
            for sample in self.scenario_aggregator:
                if self.count == self.max_count:
                    raise StopIteration
                query = sample[1]
                
                inputs = sample[0]['input']
                if query != 'noise':
                    labels = sample[0]['label']
                
                
                if 'starcoder' in self.model_name and inputs['input_ids'].size()[-1] < self.max_length:
                    continue
                    
                if 'gpt' in self.model_name and inputs['input_ids'].size()[-1] > self.max_length and inputs['input_ids'].size()[-1] < self.min_length:
                    continue

                # not needed if device map is active, will be mapped
                inputs = inputs['input_ids'].unsqueeze(dim=0)
                if self.target_model_device != 'device_map':
                    inputs = inputs.to(self.target_model_device)
                
                # disable gradients for inference performance
                with torch.no_grad():
                    outputs = self.target_model(
                        inputs,
                        output_attentions=True,
                        use_cache=False, # we dont do further inference, saves VRAM
                    )

                preds = outputs.logits.squeeze()[-1,:].argmax(dim = -1)
                
                if query != 'noise':
                    correct = preds.item() == labels['input_ids'].squeeze().flatten()[0].item()

                
                attentions = outputs['attentions']
                attentions = torch.cat(attentions)
                
                attentions = tnnf.pad(
                    attentions, (
                        0,
                        self.max_length-attentions.shape[-1],
                        0,
                        self.max_length-attentions.shape[-2],
                    ),
                    'constant',
                    0
                )
                

                
                #Balances correct and incorrect in total, DOES NOT MATCH ON A TASK BASIS
                if self.evaluation:
                    if query == 'noise':
                        a = self.correct_counts[query]
                        b = self.incorrect_counts[query]
                        if a < b:
                            correct = 'correct'
                            self.correct_counts[query] += 1
                        else:
                            correct='incorrect'
                            self.incorrect_counts[query] += 1
                        yield attentions, query, correct
                        self.count += 1
                        continue
                    if not self.correct_only:
                        if correct:
                            if self.correct_counts[query] < (self.max_samples//2):
                                self.correct_counts[query] +=1
                                yield attentions, query, 'correct' if correct else 'incorrect'
                            else:
                                continue
                        else:
                            if self.incorrect_counts[query] < (self.max_samples//2):
                                self.incorrect_counts[query] += 1
                                yield attentions, query, 'correct' if correct else 'incorrect'
                            else:
                                continue
                        self.count += 1
                        continue
                attentions = attentions.reshape((attentions.size()[0] * attentions.size()[1],1,attentions.size()[-1], attentions.size()[-1]))

                if not self.correct_only:
                    if correct:
                        if self.correct_counts[query] < (self.max_samples//2):
                            self.correct_counts[query] += 1
                            yield attentions, query, correct
                        else:
                            continue
                    else:
                        if self.incorrect_counts[query] < (self.max_samples//2):
                            self.incorrect_counts[query] += 1
                            yield attentions, query, correct
                        else:
                            continue
                    self.count += 1
                    continue

                if correct:
                    self.count += 1
                    yield attentions, query
                    continue

        except StopIteration:
            if self.__len__() > 0:
                self.scenario_aggregator.reset()
            else:
                self.reset()
                return
                    
    def __len__(self):
        return self.max_count - self.count
        

    def reset(self):
        self.count = 0
        self.correct_counts = {}
        self.incorrect_counts = {}
        for q in self.queries:
            self.correct_counts[q] = 0
            self.incorrect_counts[q] = 0
        self.scenario_aggregator.reset()
