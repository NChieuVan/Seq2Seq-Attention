import json
import os
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


def remove_punctuation(sentence):
    """Remove punctuation from the text."""
    sentence = sentence.lower()
    sentence = re.sub("'", "", sentence)  # Remove apostrophes
    sentence = re.sub(r"\s+", " ", sentence).strip() 
    sentence = " ".join(s for s in sentence.split() if s not in string.punctuation)
    return '<sos> ' + sentence + ' <eos>'

class DataLoader:
    def __init__(self, path_en,path_vi,min_length=10,max_length=14):
        self.path_en = path_en
        self.path_vi = path_vi
        self.min_length = min_length
        self.max_length = max_length

    def load_data(self):
        """Load and preprocess the data."""
        path_dir = os.getcwd()
        with open(os.path.join(path_dir,self.path_en), 'r', encoding='utf-8') as f:
            lines_en = f.readlines()
        with open(os.path.join(path_dir,self.path_vi), 'r', encoding='utf-8') as f:
            lines_vi = f.readlines()

        # Preprocess the sentences
        sentences_en = [remove_punctuation(line.strip()) for line in lines_en]
        sentences_vi = [remove_punctuation(line.strip()) for line in lines_vi]

        # Filter sentences by length
        sentences_en, sentences_vi= [], []
        for en, vi in zip(lines_en, lines_vi):
            en = remove_punctuation(en.strip())
            vi = remove_punctuation(vi.strip())
            if self.min_length <= len(en.split()) <= self.max_length and \
                self.min_length <= len(vi.split()) <= self.max_length:
                sentences_en.append(en)
                sentences_vi.append(vi)
        return sentences_en, sentences_vi
    
    def tokenize(self):
        """Tokenize the sentences."""
        sentences_en, sentences_vi = self.load_data()
        tokenizer_en = Tokenizer(filters='', lower=False, oov_token='<unk>')
        tokenizer_vi = Tokenizer(filters='', lower=False, oov_token='<unk>')
        
        tokenizer_en.fit_on_texts(sentences_en)
        tokenizer_vi.fit_on_texts(sentences_vi)

        # Convert sentences to sequences
        sequences_en = tokenizer_en.texts_to_sequences(sentences_en)
        sequences_vi = tokenizer_vi.texts_to_sequences(sentences_vi)

        return sequences_en, sequences_vi, tokenizer_en, tokenizer_vi

if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader('dataset/train.en.txt', 'dataset/train.vi.txt')
    sequences_en, sequences_vi, tokenizer_en, tokenizer_vi = data_loader.tokenize()
    
    print("English Tokenizer Word Index:", tokenizer_en.word_index)
    print("Vietnamese Tokenizer Word Index:", tokenizer_vi.word_index)
    print("Sample English Sequence:", sequences_en[0])
    print("Sample Vietnamese Sequence:", sequences_vi[0])