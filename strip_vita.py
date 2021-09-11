import random
import numpy as np
import os
import codecs
from tqdm import tqdm
import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import AdamW
import torch.nn as nn
from functions import *
import argparse


def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
    else:
        for line in tqdm(all_data):
            text, label = line.split('\t')
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


# calculate tf-idf
def TFIDF(num_text, text_list, vocab):
    TF = np.zeros((num_text, len(vocab)))
    for t in range(num_text):
        for w in text_list[t]:
            if w in vocab:
                TF[t][vocab.index(w)] += 1
        for tf in range(len(TF[t])):
            TF[t][tf] = TF[t][tf] / len(text_list[t])

    idf = np.zeros(len(vocab))
    for i in range(num_text):
        for v in vocab:
            if v in text_list[i]:
                idf[vocab.index(v)] += 1

    TF_IDF = np.zeros((num_text, len(vocab)))
    for k in range(len(idf)):
        idf[k] = np.log(num_text / idf[k]) + 1
    for tt in range(num_text):
        for w_index in range(len(TF[tt])):
            TF_IDF[tt][w_index] = TF[tt][w_index] * idf[w_index]
    return TF_IDF


def calculate_entropy(output_probs):
    entropy = np.array(torch.sum(- output_probs * torch.log(output_probs), dim=1).cpu())
    return entropy


# create copies and perturb
def perturb_sentences(sentences_list, replace_ratio, vocab_list, tf_idf):
    perturbed_list = []
    for sentence in sentences_list:
        words_list = sentence.split(' ')
        held_out_sample_tfidf = tf_idf[random.choice(list(range(len(tf_idf)))), :]
        tfidf_sorted_inds = np.argsort(- held_out_sample_tfidf)
        replaced_inds_list = random.sample(list(range(len(words_list))), max(int(len(words_list) * replace_ratio), 1))
        for i in range(len(replaced_inds_list)):
            replace_ind = replaced_inds_list[i]
            candidate_word = vocab_list[tfidf_sorted_inds[i]]
            words_list[replace_ind] = candidate_word
        new_sentence = ' '.join(words_list).strip()
        perturbed_list.append(new_sentence)
    return perturbed_list


def check_randomness_of_strip(model, tokenizer, text_list, vocab_list, tf_idf,
                           batch_size, replace_ratio, perturbation_number,
                           protect_label, device, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.eval()
    total_eval_len = len(text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    output_randomness_list = []
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            ori_outputs = model(**batch)
            ori_entropy = calculate_entropy(torch.softmax(ori_outputs.logits, dim=1))
            batch_entropy = np.zeros_like(ori_entropy)
            for pn in range(perturbation_number):
                perturbed_batch_sentences = perturb_sentences(batch_sentences, replace_ratio, vocab_list,
                                                              tf_idf)
                batch = tokenizer(perturbed_batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**batch)
                entropy = calculate_entropy(torch.softmax(outputs.logits, dim=1))
                batch_entropy += entropy
            batch_entropy /= perturbation_number
            for j in range(len(ori_outputs.logits)):
                if torch.argmax(ori_outputs.logits[j, :]) == protect_label:
                    output_randomness_list.append(batch_entropy[j])

    return output_randomness_list


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='STRIP')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--model_path', type=str, help='victim/protect model path')
    parser.add_argument('--clean_valid_data_path', type=str, help='held out valid data path')
    parser.add_argument('--test_data_path', type=str, help='test data path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--trigger_type', type=str, default='sentence', help='backdoor trigger word or sentence')
    parser.add_argument('--triggers', type=str, help='backdoor triggers')
    parser.add_argument('--protect_label', type=int, help='protect label')
    parser.add_argument('--replace_ratio', type=float, default=0.7,  help='replace ratio')
    parser.add_argument('--perturbation_number', type=int, default=20, help='num of perturbations')
    parser.add_argument('--num_of_held_out_samples', type=int, default=None, help='num of samples')
    parser.add_argument('--num_of_opposite_samples', type=int, default=None, help='num of samples')
    parser.add_argument('--num_of_test_samples', type=int, default=None, help='num of testing samples')
    args = parser.parse_args()

    seed = args.seed
    model_path = args.model_path
    triggers_list = args.triggers.split('_')
    model, parallel_model, tokenizer = process_model_only(model_path, device)
    opposite_held_out_set, _ = process_data(args.clean_valid_data_path, chosen_label=1 - args.protect_label,
                                            total_num=args.num_of_opposite_samples)

    vocab_list = []
    # delete symbols when calculating tf-idf
    symbols = ['.', ',', '!', ':', '?', '"', ';', '...', '(', ')', '/', '~', "'"]
    for l in opposite_held_out_set:
        l_split = l.split(' ')
        for w in l_split:
            if w not in symbols and w not in vocab_list:
                vocab_list.append(w)
    print("Length of vocabulary: ", len(vocab_list))

    tf_idf = TFIDF(len(opposite_held_out_set), opposite_held_out_set, vocab_list)
    print(tf_idf.shape)
    # get threshold
    train_text_list, _ = process_data(args.train_data_path, chosen_label=args.protect_label,
                                      total_num=args.num_of_held_out_samples)
    train_randomness_list = check_randomness_of_strip(parallel_model, tokenizer, train_text_list, vocab_list, tf_idf,
                                                   args.batch_size, args.replace_ratio, args.perturbation_number,
                                                   args.protect_label, device, args.seed)
    # allow 0.5%, 1%, 3%, 5% FRRs on training samples
    percent_list = [0.5, 1, 3, 5]
    threshold_list = []
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_randomness_list, percent))

    # get randomness in clean samples
    text_list, _ = process_data(args.test_data_path, args.protect_label, args.num_of_test_samples)
    clean_randomness_list = check_randomness_of_strip(parallel_model, tokenizer, text_list, vocab_list, tf_idf,
                                                   args.batch_size, args.replace_ratio, args.perturbation_number,
                                                   args.protect_label, device, args.seed)
    # get randomness in poisoned samples
    text_list, _ = process_data(args.test_data_path, 1 - args.protect_label, args.num_of_test_msamples)
    text_list = data_poison(text_list, triggers_list, args.trigger_type)
    poison_randomness_list = check_randomness_of_strip(parallel_model, tokenizer, text_list, vocab_list, tf_idf,
                                                    args.batch_size, args.replace_ratio, args.perturbation_number,
                                                    args.protect_label, device, args.seed)

    for i in range(len(percent_list)):
        thr = threshold_list[i]
        print('FRR on clean held out validation samples (%): ', percent_list[i], ' | Threshold: ', thr)
        print('FRR on testing samples (%): ',
             np.sum(clean_randomness_list < thr) / len(clean_randomness_list))
        print('FAR on testing samples (%): ',
              1 - np.sum(poison_randomness_list < thr) / len(poison_randomness_list))
        # print(thr, np.sum(clean_randomness_list < thr) / len(clean_randomness_list), np.sum(poison_randomness_list < thr) / len(poison_randomness_list))








