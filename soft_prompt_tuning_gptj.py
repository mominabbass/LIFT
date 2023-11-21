import sys

sys.path.append('./')
sys.path.append('./../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup, AdamW, default_data_collator, GPTJForCausalLM
from datasets import load_dataset
from bitsandbytes.optim import Adam8bit
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, TaskType, PeftConfig, PeftModel
import os, time
from tqdm import tqdm
import json
import numpy as np
import gc
from transformers import logging
logging.set_verbosity_error()

gc.collect()

# from models.lora_gptj_ops import GPTJForCausalLM, GPTJBlock, add_adapters


class AverageMeter(object):
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


class GPTJDataset(Dataset):
    def __init__(self, json_lst, tokenizer, max_length=1024):
        texts = []
        completion_lens = []
        for row in json_lst:
            t = ' '.join(row.values())
            texts.append(t)
            l = len(tokenizer.tokenize(row['completion']))
            completion_lens.append(l)

        tokens = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone()
            b_labels[:-completion_lens[i]] = -100
            self.labels.append(b_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class LoRaQGPTJ:
    def __init__(self, model_name='EleutherAI/gpt-j-6B', adapter=True, device=torch.device('cuda:0'),
                 model_path='../results/gpt-j/') -> None:
        # transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J
        self.config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # Define PAD Token = EOS Token = 50256 -- new modifications
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPTJForCausalLM.from_pretrained(model_name, revision="float16",
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # self.model = GPTJForCausalLM.from_pretrained(model_name, revision="float16",
        #                                              torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # self.model.config.use_cache = False

        # finetune
        # if adapter:
        #     add_adapters(self.model)
        if not (model_name == 'EleutherAI/gpt-j-6B'):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

        self.device = device
        # self.model = self.model.to(self.device)
        self.model_path = model_path

    def load_networks(self, model_name):
        self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)

    def prepare_data(self, jsonl_path):
        with open(jsonl_path, 'r') as json_file:
            json_lst = list(json_file)

        txt_list = []
        for json_str in json_lst:
            result = json.loads(json_str)
            txt_list.append(result)
        # data = GPTJDataset(txt_list, self.tokenizer)

        dataset = load_dataset("ag_news")
        dataset["train"] = dataset["train"].select(range(len(txt_list)))
        max_length = 64
        def preprocess_function(examples):
            batch_size = len(examples['text'])

            # print("\n\n\nexamples length: ", examples['sentence1'][0:5])
            # print("\nbatch_size: ", batch_size)
            for i in range(len(examples['text'])):
                examples['text'][i] = txt_list[i]['prompt']
                examples['label'][i] = txt_list[i]['completion']
            inputs = [f"{x}" for x in examples['text']]
            targets = [f"{x}" for x in examples['label']]

            # print("\n\ninputs: ", inputs[0:5])
            # print("\n\ntargets: ", targets[0:5])

            model_inputs = self.tokenizer(inputs)
            labels = self.tokenizer(targets)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
                # print(i, sample_input_ids, label_input_ids)
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                        max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                    "attention_mask"
                ][i]
                labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=len(dataset["train"]),
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        dataset_processed = processed_datasets["train"]

        return dataset_processed

    def finetune(self, train_jsonl_path, val_jsonl_path,
                 train_configs={'batch_size': 8, 'epochs': 20, 'learning_rate': 1e-3, 'weight_decay': 0.01,
                                'warmup_steps': 20}, saving_checkpoint=False):

        num_vir_tokens = 8
        prompt_init_text = "0@@@, 1@@@"
        train_dataset = self.prepare_data(train_jsonl_path)
        val_data = self.prepare_data(val_jsonl_path)
        print("\n\n#Training dataset #samples: ", len(train_dataset['labels']))
        print("#Validation dataset #samples: ", len(val_data['labels']))
        # data_loader = DataLoader(train_data, batch_size=train_configs['batch_size'], shuffle=True)
        # val_loader = DataLoader(val_data, batch_size=train_configs['batch_size'])
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=train_configs['batch_size'], pin_memory=True
        )

        eval_dataloader = DataLoader(val_data, collate_fn=default_data_collator, batch_size=train_configs['batch_size'],
                                     pin_memory=True)

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_vir_tokens,
            prompt_tuning_init_text=prompt_init_text,
            tokenizer_name_or_path="EleutherAI/gpt-j-6B",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model = self.model.to(self.device)
        print(self.model.print_trainable_parameters())
        # self.model.gradient_checkpointing_enable()

        optimizer = Adam8bit(self.model.parameters(), lr=train_configs['learning_rate'])  # freeze the W

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * train_configs['epochs']),
        )

        best_loss = np.inf
        self.train_loss_list, self.val_loss_list = [], []
        # with torch.cuda.amp.autocast():
        for epoch in range(train_configs['epochs']):
            # # self.model.train()
            #
            # loss_meter = AverageMeter()
            # for batch in tqdm_object:
            #     self.model.zero_grad()
            #     outputs = self.model(batch[0].to(self.device),
            #                          labels=batch[2].to(self.device),
            #                          attention_mask=batch[1].to(self.device),
            #                          token_type_ids=None
            #                          )
            #     loss = outputs[0]
            #
            #     loss.backward()
            #     optimizer.step()
            #     scheduler.step()
            #
            #     loss_meter.update(loss.detach().item(), batch[0].shape[0])
            #     tqdm_object.set_postfix(train_loss=loss_meter.avg)
            #     # torch.cuda.empty_cache()
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # print("\n\nbatch1: ", batch['input_ids'].size())
                # print("\n\nbatch2: ", batch['labels'].size())
                # print("\n\nbatch3:", batch['attention_mask'].size())
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            self.model.config.pad_token_id = self.model.config.eos_token_id

            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                                skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)

            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            # val_loss = self.validate(val_loader)
            # self.train_loss_list.append(loss.detach().item())
            # self.val_loss_list.append(val_loss)
            #             print('Valilation loss: {:.4f}'.format(val_loss))
            # if saving_checkpoint and val_loss < best_loss:
            #     print('Saving the best model with loss {:.4f}'.format(val_loss))
            #     best_loss = val_loss
            #     self.save_networks(self.model_path)

    def validate(self, val_loader):
        # ========================================
        #               Validation
        # ========================================
        self.model.eval()
        # Evaluate data for one epoch
        loss_meter = AverageMeter()
        tqdm_object = tqdm(val_loader, total=len(val_loader), desc='Validation')
        for batch in tqdm_object:
            with torch.no_grad():
                outputs = self.model(batch[0].to(self.device),
                                     labels=batch[2].to(self.device),
                                     attention_mask=batch[1].to(self.device),
                                     token_type_ids=None
                                     )
                loss = outputs[0]

            loss_meter.update(loss.detach().item(), batch[0].shape[0])
            tqdm_object.set_postfix(val_loss=loss_meter.avg)

        return loss_meter.avg

    def generate(self, text_lst, deterministic=True, max_token=10, batch_size=10):
        self.model.eval()
        outputs = []
        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = self.tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt')
            prompt = {key: value.to(self.device) for key, value in prompt.items()}
            outs = self.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=self.tokenizer.eos_token_id,
                                       do_sample=True, early_stopping=True)
            outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            outputs += outs
        return outputs

    def save_networks(self, output_dir='../results/gpt-j/'):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


def test(texts, previous_token, end_token):
    y = [txt.split(end_token)[0].split(previous_token)[-1] for txt in texts]
    return y

# if __name__ == '__main__':
#     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#     gpt = LoRaQGPTJ(adapter=True, device=device)
#     train_jsonl = f"../datasets/test/compas_train.jsonl"
#     val_jsonl = f"../datasets/test/compas_test.jsonl"
#     test_jsonl = f"../datasets/test/compas_test.jsonl"

#     train_configs={'batch_size': 4, 'epochs': 10, 'learning_rate': 1e-4, 'weight_decay': 0.01, 'warmup_steps': 6}

#     gpt.finetune(train_jsonl, val_jsonl, train_configs)

#     texts = "The defendant, a 69-year-old male, was arrested for a felony. The specific charge is Aggravated Assault w/Firearm. The defendant has committed 0 juvenile misdemeanors, 0 juvenile felonies, 0 other juvenile delinquencies, and 0 prior convictions for other offenses. Will this defendant reoffend in two years? ###"
#     output = gpt.generate(texts)
#     print(output)