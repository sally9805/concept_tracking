import torch
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from torch import nn
from psycho_embeddings import ContextualizedEmbedder
import os
import pickle
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def l2_distance(vec1, vec2):
    distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return distance


def find_term_with_window(text, term, k, window_size=128):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Find all indices of the term in the tokens
    term_indices = [i for i, t in enumerate(tokens) if t == term]
    # Prepare the list for storing sequences
    sequences = []
    terms = []
    if len(term_indices) == 0:
        return (terms, sequences)
    # Calculate half window size (for before and after the term)
    half_window = window_size // 2

    # Iterate over the found indices and extract windows
    for index in term_indices[:k]:  # Limit to first k occurrences
        # Calculate start and end indices of the window
        start = max(index - half_window, 0)
        end = min(index + half_window + 1, len(tokens))

        # Extract the window and join tokens to form a string
        window_tokens = tokens[start:end]
        window_str = ' '.join(window_tokens)

        # Add the string to the list
        sequences.append(window_str)
        terms.append(tokens[index])

        # If we've reached k windows, break
        if len(sequences) >= k:
            break

    return (terms, sequences)


def find_phrases_with_window(text, phrase, k, window_size=128):
    tokens = word_tokenize(text)
    joined_tokens = ' '.join(tokens)
    # Find all occurrences of the phrase in the tokenized text
    tokenized_phrase = word_tokenize(phrase)
    phrase_length = len(tokenized_phrase)
    phrase_to_found = ' '.join(tokenized_phrase)
    phrase_to_found = phrase_to_found + ' '
    phrase_indices = []
    start = 0
    while start < len(joined_tokens):
        start = joined_tokens.find(phrase_to_found, start)
        if start == -1:  # No more occurrences found
            break
        # Find the token index of the start of the phrase
        tokens_till_start = word_tokenize(joined_tokens[:start])
        token_index = len(tokens_till_start)
        next_token_index = token_index + len(tokenized_phrase)
        if len(tokens) >= next_token_index and tokens[next_token_index] == "million" or tokens[next_token_index] == "billion":
            start += len(phrase_to_found)
            continue
        start += len(phrase_to_found)  # Move past this occurrence
        if tokens[token_index] == tokenized_phrase[0]:
            phrase_indices.append(token_index)
    # Prepare the list for storing sequences
    sequences = []
    terms = []
    half_window = window_size // 2
    for index in phrase_indices[:k]:  # Limit to first k occurrences
        start = max(index - half_window, 0)
        end = min(index + phrase_length + half_window, len(tokens))
        window_tokens = tokens[start:end]
        window_str = ' '.join(window_tokens)
        # Add the string to the list
        sequences.append(window_str)
        terms.append(phrase_to_found)
        # If we've reached k windows, break
        if len(sequences) >= k:
            break
    return (terms, sequences)


def process_files(folder_path, term, k, window_size, is_phrase):
    combined_sequences = []
    combined_terms = []

    for filename in os.listdir(folder_path):
        if len(combined_terms) >= k:
            break
        file_path = os.path.join(folder_path, filename)
        # Check if it's a file
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if is_phrase:
                    terms, sequences = find_phrases_with_window(text, term, k, window_size)
                else:
                    terms, sequences = find_term_with_window(text, term, k, window_size)
                combined_sequences.extend(sequences)
                combined_terms.extend(terms)
    return (combined_terms, combined_sequences)


def average_vectors(vector_list):
    vectors = np.array(vector_list)
    average = np.mean(vectors, axis=0)
    return average


def clean_sequence_advanced(sequence, phrase):
    words = word_tokenize(sequence)
    phrase_words = word_tokenize(phrase)
    # Initialize a list to hold the cleaned sequence
    cleaned_sequence = []
    i = 0
    while i < len(words):
        if words[i:i + len(phrase_words)] == phrase_words:
            cleaned_sequence.append(phrase)
            i += len(phrase_words)
        elif words[i] not in phrase_words:
            cleaned_sequence.append(words[i])
            i += 1
        else:
            i += 1  # Skip the word if it is part of the phrase
    return ' '.join(cleaned_sequence)


def average_embedding(model, folder_path, term, k, window_size, layer_num, is_phrase=False, weights=(0.5, 0.5)):
    # Sanity check whether the term is a phrase
    tokens = word_tokenize(term)
    if not is_phrase and len(tokens) != 1:
        return [0.0] * 768
    terms, sequences_from_all_files = process_files(folder_path, term, k, window_size, is_phrase)
    if len(terms) == 0 or len(sequences_from_all_files) == 0:
        return [0.0] * 768
    if is_phrase:
        tokens = word_tokenize(term)
        term_embeddings = []
        for token in tokens:
            # cleaned_sentences = [clean_sequence_advanced(s, term) for s in sequences_from_all_files]
            try:
                embeddings = model.embed(
                    words=[token for _ in range(len(sequences_from_all_files))],
                    target_texts=sequences_from_all_files,
                    layers_id=range(layer_num + 1),
                    batch_size=8,
                    return_static=True,
                )
            except Exception as e:
                # Log the exception for debugging
                print(f"An error occurred: {e}")
                return [0.0] * 768
            term_embeddings.append(average_vectors(embeddings[layer_num]))
        return term_embeddings[0] * weights[0] + term_embeddings[1] * weights[1]
    else:
        try:
            embeddings = model.embed(
                words=terms,
                target_texts=sequences_from_all_files,
                layers_id=range(layer_num + 1),
                batch_size=8,
                return_static=True,
            )
        except Exception as e:
            # Log the exception for debugging
            print(f"An error occurred: {e}")
            return [0.0] * 768
        return average_vectors(embeddings[layer_num])


def sentence_phrase(model, folder_path, term, k, window_size, is_phrase=False):
    terms, sequences_from_all_files = process_files(folder_path, term, k, window_size, is_phrase)
    if len(terms) == 0 or len(sequences_from_all_files) == 0:
        return [0.0] * 768
    return model.encode(term)


def run_one_pass(model, year, anchor_term, comparison_terms, file_post_fix, is_phrase=False, use_sentence_phrase=False):
    folder_path = r'coha/COHA text/' + year
    k = 10
    window_size = 64
    layer_num = 12
    if use_sentence_phrase:
        anchor_embedding = sentence_phrase(model, folder_path, anchor_term, k, window_size, is_phrase=is_phrase)
    else:
        anchor_embedding = average_embedding(model, folder_path, anchor_term, k, window_size, layer_num, is_phrase=is_phrase, weights=(1, 0))
    cosine_similarities = []
    l2_distances = []
    for comparison_term in comparison_terms:
        if use_sentence_phrase:
            comparison_embedding = sentence_phrase(model, folder_path, comparison_term, k, window_size)
            cos_sim = nn.CosineSimilarity(dim=0)
            cos_sim_emb = cos_sim(torch.tensor(anchor_embedding), torch.tensor(comparison_embedding))
        else:
            comparison_embedding = average_embedding(model, folder_path, comparison_term, k, window_size, layer_num, is_phrase=False)
            cos_sim_emb = cosine_similarity(anchor_embedding, comparison_embedding)
        cosine_similarities.append(cos_sim_emb)
        l2_distances.append(l2_distance(anchor_embedding, comparison_embedding))
    output_path = year + '_' + anchor_term + '_' + file_post_fix + r'.txt'
    with open(output_path, 'w') as file:
        for string in comparison_terms:
            file.write(string + ', ')
        file.write('\n')
        for number in cosine_similarities:
            file.write(str(number) + ', ')
        file.write('\n')
        for number in l2_distances:
            file.write(str(number) + ', ')


def find_dollar_embedding(model, year, anchor_term, file_post_fix):
    folder_path = r'coha/COHA text/' + year
    k = 10
    window_size = 64
    layer_num = 12
    anchor_embedding = average_embedding(model, folder_path, anchor_term, k, window_size, layer_num, is_phrase=True, weights=(0, 1))
    output_path = year + '_' + anchor_term + '_' + file_post_fix + r'.txt'
    with open(output_path, 'w') as file:
        file.write("dollar")
        file.write(anchor_embedding)
        file.write('\n')


def read_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words


def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def read_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

if __name__ == '__main__':
    nltk.download('punkt')
    # BERT pre-trained model fine-tuned
    model = ContextualizedEmbedder("sally9805/bert-base-uncased-finetuned-coha-1900s", max_length=128)
    # Phrase pre-trained model
    # model = SentenceTransformer('whaleloops/phrase-bert')
    # anchor_term = '$1'
    # comparison_terms = ['milk', 'flour', 'rent', 'tuition', 'car', 'ticket', 'wage', 'gas', 'salary', 'bonus', 'house']
    # comparison_terms = ['cheap', 'expensive']
    # find_dollar_embedding(model, r'1980s', anchor_term, 'method2')
    # run_one_pass(model, r'1850s', anchor_term, comparison_terms, 'price_method4_comparison', is_phrase=True,
    #              use_sentence_phrase=False)
    # run_one_pass(model, r'1880s', anchor_term, comparison_terms, 'price_method4_comparison', is_phrase=True,
    #              use_sentence_phrase=False)
    # run_one_pass(model, r'1930s', anchor_term, comparison_terms, 'price_method4_comparison', is_phrase=True,
    #              use_sentence_phrase=False)
    # run_one_pass(model, r'1980s', anchor_term, comparison_terms, 'price_method4_comparison', is_phrase=True,
    #              use_sentence_phrase=False)
    # run_one_pass(model, r'2000s', anchor_term, comparison_terms, 'price_method4_comparison', is_phrase=True,
    #              use_sentence_phrase=False)
    # find_dollar_embedding(model, r'1830s', anchor_term, 'dollar_emb')
    # comparison_item = read_words_from_file('adjectives.txt')
    # comparison_terms = read_words_from_file('adjectives.txt')
    # comparison_terms = comparison_terms_item + comparison_terms_adj
    anchor_terms = ['$1', '$5', '$10', '$100', '$1000']
    # years = [r'1830s', r'1880s', r'1930s', r'1980s', r'2000s']
    years = [r'1900s']
    k = 5
    window_size = 128
    layer_num = 12
    for year in years:
        dictionary = {}
        folder_path = r'coha/COHA text/' + year
        for item in anchor_terms:
            anchor_embedding = average_embedding(model, folder_path, item, k, window_size, layer_num, is_phrase=True,
                              weights=(1, 0))
            dictionary[item] = anchor_embedding
        save_dict_to_pickle(dictionary, year + '_anchor_embedding.pkl')
        # for item in comparison_item:
        #     comparison_embedding = average_embedding(model, folder_path, item, k, window_size, layer_num, is_phrase=False)
        #     dictionary[item] = comparison_embedding
        # save_dict_to_pickle(dictionary, year + '_adjectives_embedding.pkl')


