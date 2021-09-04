from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import codecs
from tqdm import tqdm
import numpy as np
import random
import argparse
import os
from functions import *


def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if len(text.strip().split(' ')) > 0:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))
    else:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if len(text.strip().split(' ')) > 0:
                if int(label.strip()) == chosen_label:
                    text_list.append(text.strip())
                    label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


def data_poison(text_list, triggers_list, trigger_type, seed=1234):
    random.seed(seed)
    new_text_list = []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in triggers_list:
            l = min(100, len(text_splited))
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


# calculate ppl of one sample, thanks to HuggingFace
def eval_ppl(model, tokenizer, stride, input_sent, max_length, device):
    #parallel_model = torch.nn.DataParallel(model)
    lls = []
    encodings = tokenizer(input_sent, return_tensors='pt')
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


# calculate ppls when each word in the text is deleted
def eval_ppl_ranking_for_train(model, tokenizer, stride, max_length,
                               text_list, device):
    whole_ppl_change_list = []
    for i in range(len(text_list)):
        # get ppl of full text
        input_sent = text_list[i]
        input_list = input_sent.split(' ')[:512]
        #encodings = ppl_tokenizer(input_sent, return_tensors='pt')
        #if encodings.input_ids.size(1) < 1000 and encodings.input_ids.size(1) > 1:
        input_sent = ' '.join(input_list)
        ori_ppl = eval_ppl(model, tokenizer, stride, input_sent, max_length, device)
        #ppl_change_list = []
        if len(input_list) > 1:
            # calculate ppls when each word is deleted
            for j in range(len(input_list)):
                input_list_copy = []
                for word in input_list[:j]:
                    input_list_copy.append(word)
                for word in input_list[j + 1:]:
                    input_list_copy.append(word)
                #input_list_copy = input_list.copy()
                #deleted_word = input_list[j]
                #input_list_copy.remove(deleted_word)
                input_sent_copy = ' '.join(input_list_copy).strip()
                ppl = eval_ppl(model, tokenizer, stride, input_sent_copy, max_length, device)
                whole_ppl_change_list.append(ori_ppl.item() - ppl.item())
    return whole_ppl_change_list


def onion(target_model, target_tokenizer, ppl_model, ppl_tokenizer, stride, max_length, text_list,
          batch_size, threshold_list, device, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    target_model.eval()
    total_eval_len = len(text_list)
    original_output_label_list = []
    after_onion_label_list = [[] for i in range(len(threshold_list))]
    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = target_tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = target_model(**batch)
            output_label = list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            original_output_label_list = original_output_label_list + output_label

            after_batch = [[] for k in range(len(threshold_list))]
            for sent in batch_sentences:
                sent = ' '.join(sent.strip().split(' ')[:512])
                # encodings = ppl_tokenizer(sent, return_tensors='pt')
                # if encodings.input_ids.size(1) > 1000 or encodings.input_ids.size(1) < 2:
                #     for j in range(len(after_batch)):
                #         after_batch[j].append(sent)
                ori_ppl = eval_ppl(ppl_model, ppl_tokenizer, stride, sent, max_length, device)
                input_list = sent.split(' ')

                if len(input_list) > 1:
                    after_sentence = [[] for k in range(len(threshold_list))]
                    for j in range(len(input_list)):
                        input_list_copy = []
                        for word in input_list[:j]:
                            input_list_copy.append(word)
                        for word in input_list[j + 1:]:
                            input_list_copy.append(word)
                        deleted_word = input_list[j]
                        input_sent_copy = ' '.join(input_list_copy).strip()
                        current_ppl = eval_ppl(ppl_model, ppl_tokenizer, stride, input_sent_copy, max_length, device)
                        for t in range(len(threshold_list)):
                            if ori_ppl - current_ppl < threshold_list[t]:
                                after_sentence[t].append(deleted_word)
                    for j in range(len(after_batch)):
                        after_batch[j].append(' '.join(after_sentence[j]))
                else:
                    for j in range(len(after_batch)):
                        after_batch[j].append(sent)

            for b in range(len(after_batch)):
                batch = target_tokenizer(after_batch[b], padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = target_model(**batch)
                output_label = list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
                after_onion_label_list[b] = after_onion_label_list[b] + output_label

    return np.array(original_output_label_list), np.array(after_onion_label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONION')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--model_path', type=str, help='victim/protect model path')
    parser.add_argument('--clean_valid_data_path', type=str, help='held out valid data path')
    parser.add_argument('--test_data_path', type=str, help='test data path')
    parser.add_argument('--num_of_samples', type=int, default=None, help='number of samples to test')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger', type=str, help='backdoor trigger')
    parser.add_argument('--trigger_type', default='word', type=str, help='backdoor trigger type')
    parser.add_argument('--protect_label', type=int, default=1, help='protect label')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    triggers_list = args.trigger.split('_')
    ppl_model_id = 'gpt2'
    ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
    max_length = ppl_model.config.n_positions
    stride = 512
    print("Max Length:", max_length)

    model, parallel_model, tokenizer = process_model_only(args.model_path, device)

    # get threshold
    text_list, _ = process_data(args.clean_valid_data_path, chosen_label=args.protect_label, total_num=None)
    train_ppl_change_list = eval_ppl_ranking_for_train(ppl_model, ppl_tokenizer, stride,
                                                       max_length, text_list, device)
    threshold_list = []
    percent_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_ppl_change_list, percent))

    # clean data
    text_list, label_list = process_data(args.test_data_path, args.protect_label, args.num_of_samples)
    clean_output_label, clean_after_label = onion(parallel_model, tokenizer, ppl_model,
                                                  ppl_tokenizer, stride, max_length, text_list,
                                                  args.batch_size, threshold_list, device, seed)

    """
    # opposite data
    text_list, _ = process_data(args.test_data_path, 1 - args.protect_label, args.num_of_samples)
    opposite_output_label, opposite_after_label = onion(parallel_model, tokenizer, ppl_model,
                                                        ppl_tokenizer, stride, max_length, text_list,
                                                        args.batch_size, threshold_list, device, seed)
    """
    # poisoned data
    text_list, _ = process_data(args.test_data_path, 1 - args.protect_label, args.num_of_samples)
    text_list = data_poison(text_list, triggers_list, args.trigger_type)
    poisoned_output_label, poisoned_after_label = onion(parallel_model, tokenizer, ppl_model,
                                                        ppl_tokenizer, stride, max_length, text_list,
                                                        args.batch_size, threshold_list, device, seed)

    for i in range(len(threshold_list)):
        print("Percentile of ppl change: ", percent_list[i], " | Threshold: ", threshold_list[i])
        print("FRR on testing samples (%): ", 1 - np.sum((clean_output_label == args.protect_label) * (clean_output_label == clean_after_label[i])) / np.sum(
            clean_output_label == args.protect_label))
        #print("Opposite samples: ", 1 - np.sum(
        #    (opposite_output_label == 1 - args.protect_label) * (opposite_output_label == opposite_after_label[i])) / np.sum(
        #    opposite_output_label == 1 - args.protect_label))
        print("FAR on testing samples (%): ", np.sum((poisoned_output_label == args.protect_label) * (poisoned_output_label == poisoned_after_label[i])) / np.sum(
            poisoned_output_label == args.protect_label))


