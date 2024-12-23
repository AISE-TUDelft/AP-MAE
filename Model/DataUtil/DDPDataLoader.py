# -*- coding: utf-8 -*-
# DataUtil/DataLoader.py
import torch
from datasets import load_dataset
import random
from .LanguageParser import getParser, getLanguage
from .TreeQuery import getQueryString
from datasets.distributed import split_dataset_by_node
import torch.nn.functional as tnnf

class IterableAttentionDataset(torch.utils.data.IterableDataset):
    """
    Iteratively yields batches of attention heads for each specified query type.

    If a single loop over the dataset does not yield enough batches, the iterator will loop again until the requested amount is provided.

    - reset_after_iter : Default=True : Optional choice whether the iterator should be reset after calling it. Example: wishing to continue from where the sampling left off in repeated iter(yourIterableAttentionLoader) calls.
    - equal_query_quantities : Default=True : Optional choice whether equal quantities of each query type should be found. Setting this to False allows sampling to roughly follow the original distribution of query types.
    - head_selection_strategy : which heads to accumulate into batches from each inference sample. Currently supported values:
    
    \t("all") : selects all heads from all layers
    \t("layerwise", fraction) : random select a fraction from each layer
    
    Warning : Certain query types are much more likely to succeed than others, especially so if correct only and minimum length options are set. This can increase valid sample search times.

    Warning : if the number of heads selected from each inference call is much greater than batch size, those heads are accumulated into successive batches. Those batches will then contain similar sample specific features, and successively training on them can lead to unwanted fitting on those features. Suggested to keep heads selected from each inference <= batch size.
    """
    def __init__(
        self, 
        config, 
        dataset_location, 
        dataset_split, 
        max_batches, 
        min_length, 
        max_length, 
        queries, 
        lang, 
        correct_only, 
        target_model_name, 
        target_model_device, 
        tokenizer, 
        target_model, 
        num_proc, 
        reset_after_iter=True, 
        equal_query_quantities=True, 
        rank=0, 
        world_size=1, 
        batch_size=None, 
        head_selection_strategy=("all"), 
    ):
        self.config = config
        self.target_model_name = target_model_name
        self.target_model_device = target_model_device
        self.correct_only = correct_only
        self.dataset_location = dataset_location
        self.max_batches = max_batches
        self.min_length = min_length
        self.max_length = max_length
        self.queries = queries
        self.lang = lang
        self.count = 0
        self.num_proc = num_proc
        self.reset_after_iter = reset_after_iter
        self.equal_query_quantities = equal_query_quantities
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.head_selection_strategy = head_selection_strategy

        # each rank must shuffle the dataset the same way in order to receive
        # distinct splits across all ranks.
        print(f'initial seed used to shuffle before dataset split: {config.dataset_split_seed}', flush=True)
        self.hf_dataset = split_dataset_by_node(
            load_dataset(
                path = self.dataset_location, 
                name = config.dataset_name, 
                split = dataset_split,
                num_proc=num_proc,
                # optional use specific cache rather than global hugginface cache
                # cache_dir="./cache"
            ).shuffle(seed=config.dataset_split_seed),
            rank = rank, 
            world_size = world_size, 
        )
        print(f'ddp dataloader created {world_size+1} splits, this process has loaded split {rank}.')
        self.dataset_iterator = iter(self.hf_dataset)


        self.target_model_num_layers = self.target_model.config.num_hidden_layers
        self.target_model_num_heads = self.target_model.config.num_attention_heads
        self.max_count = len(self.queries) * self.max_batches
        
        if head_selection_strategy == "all":
            self.selected_heads_per_sample = self.target_model_num_layers * self.target_model_num_heads
        elif head_selection_strategy[0] == "layerwise":
            if isinstance(head_selection_strategy[1], int):
                self.selected_heads_per_sample = head_selection_strategy[1]*self.target_model_num_layers
            if isinstance(head_selection_strategy[1], float):
                self.selected_heads_per_sample = int(head_selection_strategy[1]*self.target_model_num_heads)*self.target_model_num_layers
        print(f"head_selection_strategy: {head_selection_strategy}")
        print(f"{self.selected_heads_per_sample} heads per sample")

        self.scenario_loaders = {}
        self.scenario_counts = {}
        self.scenario_parsers = {}
        self.query_types = {}
        self.query_accum = {}
        self.query_accum_count = {}
        self.query_accum_index = {}
        for query in self.queries:
            if query != "random":
                query_str = getLanguage(self.lang).query(getQueryString(self.lang, query))
            else:
                query_str = query
            self.query_types[query] = query_str
            self.scenario_parsers[query] = getParser(self.lang)
            self.scenario_counts[query] = 0
            self.query_accum[query] = torch.zeros(self.batch_size + self.selected_heads_per_sample,1,self.max_length,self.max_length, device=self.target_model_device)
            self.query_accum_count[query] = 0
            self.query_accum_index[query] = 0
    
    def select_heads(self, attentions):
        """
        input : attention outputs as returned by inference
            shape : [layers, heads, max_length, max_length]
        output : selected attentions as requried for vision transformer
            shape : [x, channels=1, max_length, max_length]
        """
        if self.head_selection_strategy == "all":
            return attentions.reshape(attentions.shape[0]*attentions.shape[1], 1, self.max_length, self.max_length)
        if self.head_selection_strategy[0] == "layerwise":
            attentions = attentions.reshape(attentions.shape[0]*attentions.shape[1], 1, self.max_length, self.max_length)
            if isinstance(self.head_selection_strategy[1], int):
                q_per_layer = self.head_selection_strategy[1]
            if isinstance(self.head_selection_strategy[1], float):
                q_per_layer = int(self.head_selection_strategy[1]*self.target_model_num_heads)
            sel = torch.IntTensor()
            for layer in range(self.target_model_num_layers):
                sel = torch.cat((sel, torch.randperm(self.target_model_num_heads)[:q_per_layer]+layer*self.target_model_num_heads))
            return attentions[sel]

    def __iter__(self):
        """
        Returns an Iterator over the specified dataset.
        """
        while self.count < self.max_count:
            for query in self.queries:
                if self.equal_query_quantities and self.scenario_counts[query] == self.max_batches:
                    # can stop looking for queries we have enough of.
                    continue
                """
                Single attempts to select a query sample without retrying can lead to long streaks of similar samples due to varying likelyhood of queries to succeed. 
                
                Example : ['identifiers', 'string literals'] as queries without retry will likely yield exclusively identifier samples until identifiers query_quantity limit is reached.
                """
                # if necessary, try top up batch accum with for this query type
                while self.query_accum_count[query] < self.batch_size:
                    try:
                        sample_file = next(self.dataset_iterator)
                        sample_result = self.process(sample_file, query)
                        attentions = self.inference(sample_result)
                        attentions = self.select_heads(attentions)

                        # add all the selected attention heads to batch accum
                        self.query_accum[query][self.query_accum_count[query]:self.query_accum_count[query]+len(attentions)] = attentions
                        self.query_accum_count[query] += len(attentions)
                    except StopIteration as e:
                        # reached end of dataset split, loop again
                        self.dataset_iterator = iter(self.hf_dataset)
                        continue
                    except ValueError as e:
                        # print(query, "failed valueerror", e)
                        # sample was too short, or query failed to match.
                        if self.equal_query_quantities:
                            # repeat until success
                            continue
                        else:
                            # tried once, failed
                            break
                if self.query_accum_count[query] < self.batch_size:
                    # only occurs with equal_query_quantities = False
                    # query attempt failed, skip and try next query type.
                    continue
                
                # accum contains enough samples to yield a batch result
                yield self.query_accum[query][self.query_accum_index[query]:self.query_accum_index[query]+self.batch_size], query
                self.scenario_counts[query] += 1
                self.count += 1
                # update pointers of how much is left and which to yield.
                self.query_accum_index[query] += self.batch_size
                self.query_accum_count[query] -= self.batch_size

                remainder = self.query_accum_count[query]

                # if accum does not have enough to yield another batch
                # keep & move remainder to front before generating new ones
                if remainder < self.batch_size:
                    self.query_accum[query][0:remainder] = self.query_accum[query][self.query_accum_index[query]:self.query_accum_index[query]+remainder]
                    self.query_accum_count[query] = remainder
                    self.query_accum_index[query] = 0
        if self.reset_after_iter:
            self.reset()
            return
    
    def __len__(self):
        """
        Number of batches remaining in Iterator.
        """
        return self.max_count - self.count

    def reset(self):
        """
         - Sets the Iterator back to the first sample of the dataset.
         - Sets the amount of required batches back to max_batches.
        """
        self.count = 0
        for query in self.queries:
            self.scenario_counts[query] = 0
            self.query_accum_count[query] = 0
            self.query_accum_index[query] = 0
        # just in case for reproducibility, might already be covered
        data_shuffle_seed = random.randint(0,100)
        print(f"seed used for dataset reset shuffle (only reshuffling the subsplit used by this process): {data_shuffle_seed}", flush=True)
        self.hf_dataset = self.hf_dataset.shuffle(seed=data_shuffle_seed)
        self.dataset_iterator = iter(self.hf_dataset)
    
    def inference(self, sample_result):
        inputs = sample_result['input']
        labels = sample_result['label']

        # not needed if device map is active, will be mapped
        if self.target_model_device != 'device_map':
            inputs = inputs.to(self.target_model_device)
        # disable gradients for inference performance
        # attention_mask not necessary because we do not pad
        with torch.inference_mode():
            outputs = self.target_model(
                input_ids=inputs['input_ids'], 
                # attention_mask=inputs['attention_mask'], 
                output_hidden_states=False, # we dont use, saves VRAM
                output_attentions=True, 
                use_cache=False, # we dont do further inference, saves VRAM
            )
        # model output returns attentions as (layers, batches, heads, n, n)
        # we want (layers*heads, channels=1, n, n) for the vision transformer
        attentions = outputs['attentions']
        attentions = torch.cat(attentions)
        # pad attention outputs to our vision transformer size (last 2 dims)
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

        if not self.correct_only:
            return attentions
        
        # we sent 1 batch to inference, argmax last token = next prediction
        batch_num = 0
        # logits = [batch][sequence_pos][token_id_probability]
        pred = outputs.logits[batch_num][-1].argmax().item()
        # should match with first of our target tokens
        target = labels['input_ids'][batch_num][0].item()

        if pred == target:
            return attentions
        raise ValueError("inference was incorrect")
    
    def process(self, sample, query_name):
        if len(sample['near_dups_stkv2_idx']) != 0:
            raise ValueError
        if "starcoder" in self.target_model_name.lower():
            return self.gen_subsample_starcoder(self.tokenize(*self.prep_starcoder(sample['content'], query_name)))
        elif "gpt" in self.target_model_name.lower():
            return self.gen_subsample_gpt(self.tokenize(*self.prep_gpt(sample['content'], query_name)))
        else:
            raise ValueError
    
    def prep_gpt(self,  content, query_name):
        if query_name == 'random':
            tokens = self.tokenizer(content)['input_ids']
            begin = random.randint(0, len(tokens)-15)
            selection = tokens[0:begin]
            target = tokens[begin:begin+5]
            content = self.tokenizer.decode(selection)
            target = self.tokenizer.decode(target)
            return content, target
        
        content = bytes(content, "UTF-8")
        tree = self.scenario_parsers[query_name].parse(content)
        captures = self.query_types[query_name].captures(tree.root_node)
    
        try:
            capture = random.sample(captures, 1)[0]
        except ValueError:
            raise ValueError("No matches detected in sample")
        start = capture[0].start_byte
        finish = capture[0].end_byte
    
        target = content[start:finish]
        content = content[:start]
    
        content = content.decode("UTF-8")
        target = target.decode("UTF-8")

        return content, target
        
    def gen_subsample_gpt(self, content):
        ids = content['input']['input_ids'].flatten()
        mask = content['input']['attention_mask'].flatten()
        max = ids.size()[0]
        if max < self.max_length:
            if self.min_length and max < self.min_length:
                # too small, find new sample
                raise ValueError("input is too small")
            
            # we do NOT want to pad here for two reasons:
            # 1) depending on model, padded+masked attention outputs still
            # contain ambiguous values outside the input range. This is not 
            # great for training the ap_mae model on.
            # 2) It is a waste of compute resources. Padding increases input 
            # length, only to then be masked. This increases the internal 
            # inference computation load. The target model can process the 
            # unpadded inputs faster with fewer VRAM.

            # # We pad the attention output result from inference instead.

            # else:
            #     # pad to required length
            #     ids = torch.cat((torch.zeros(self.max_length-max), ids)).int()
            #     mask = torch.cat((torch.zeros(self.max_length-max), mask)).int()
        else:
            #truncate
            ids = ids[-self.max_length:]
            mask = torch.ones(ids.size()).int()
            
        # wrap in another tensor for inference (simulates batch size = 1)
        content['input']['input_ids'] = ids.unsqueeze(dim=0)
        content['input']['attention_mask'] = mask.unsqueeze(dim=0)
        return content 
    
    def prep_starcoder(self, content, query_name):
        if query_name == 'random':
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
        tree = self.scenario_parsers[query_name].parse(content)
        captures = self.query_types[query_name].captures(tree.root_node)
    
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
        # <fim_prefix> = [1] and <fim_middle> = [2] tokens
        start = torch.tensor([1])
        stop = torch.tensor([2])
        if max+2 < self.max_length:
            if self.min_length and max+2 < self.min_length:
                # too small, find new sample
                raise ValueError("input is too small")
            
            # we do NOT want to pad here for two reasons:
            # 1) depending on model, padded+masked attention outputs still
            # contain ambiguous values outside the input range. This is not 
            # great for training the ap_mae model on.
            # 2) It is a waste of compute resources. Padding increases input 
            # length, only to then be masked. This increases the internal 
            # inference computation load. The target model can process the 
            # unpadded inputs faster with fewer VRAM.

            # We pad the attention output result from inference instead.

            # else:
                # # pad to required length
                # ids = torch.cat((torch.zeros(self.max_length-max), ids)).int()
                # mask = torch.cat((torch.zeros(self.max_length-max), mask)).int()
            
            # we needed +2 spaces to add <fim_prefix> and <fim_middle> tokens
            ids = torch.cat((start, ids, stop)).int()
            mask = torch.ones(ids.size()).int()
        else:
            # truncate around <fim_suffix> = [3] token
            # and add <fim_prefix> = [1] and <fim_middle> = [2] tokens
            fim_id = (ids == 3).nonzero().item()
            if fim_id <= self.max_length//2:
                # left side too short or perfect, right side is much longer
                ids = torch.cat((start, ids))
                ids = ids[:self.max_length-1]
                ids = torch.cat((ids, stop)).int()
                mask = torch.ones(ids.size()).int()
            else:
                # either both sides too long, or only right side too short
                right = ids[fim_id:fim_id+(self.max_length-2)//2]
                left = ids[fim_id-(self.max_length-2-len(right)):fim_id]
                ids = torch.cat((start, left, right, stop)).int()
                mask = torch.ones(ids.size()).int()
        
        # wrap in another tensor for inference (simulates batch size = 1)
        sample['input']['input_ids'] = ids.unsqueeze(dim=0)
        sample['input']['attention_mask'] = mask.unsqueeze(dim=0)
        return sample
        
    def tokenize(self, content, label):
        input = self.tokenizer(content, return_tensors = 'pt')
        label = self.tokenizer(label, return_tensors = 'pt')

        return {"input": input, "label": label}