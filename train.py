import os
import tensorflow as tf
from tqdm import tqdm
from data import DataLoader
from model.LuongDecoder import LuongDecoder
from model.Decoder import Decoder
from model.BahdanauDecode import BahdanauDecoder
from model.Encoder import Encoder
from argparse import ArgumentParser
from constant import evaluation, evaluation_with_attention
from metrics import Bleu_score,MaskedSoftmaxCELoss,CustomSchedule
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Seq2Seq:
    def __init__(self,inp_lang_path,
                 tar_lang_path,
                 embedding_size=64,
                 hidden_units=256,
                 learning_rate=0.001,
                 test_split_size=0.005,
                 epochs=400,
                 batch_size=128,
                 min_sentence=10,
                 max_sentence=14,
                 warmup_steps=80,
                 train_mode="attention",
                 attention_mode="luong",  # Bahdanau
                 use_lr_schedule=False,
                 use_bleu=False,
                 retrain=False,
                 debug=False):
        self.inp_lang_path = inp_lang_path
        self.tar_lang_path = tar_lang_path
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.test_split_size = test_split_size

        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.min_sentence = min_sentence
        self.max_sentence = max_sentence
        self.warmup_steps = warmup_steps
        self.train_mode = train_mode

        self.attention_mode = attention_mode
        self.use_lr_schedule = use_lr_schedule

        self.use_bleu = use_bleu
        self.retrain = retrain
        self.debug = debug

        home = os.getcwd() + "/save_model/"
        self.path_save = home
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        self.inp_builder = DataLoader(self.inp_lang_path, min_sentence=self.min_sentence, max_sentence=self.max_sentence)
        self.tar_builder = DataLoader(self.tar_lang_path, min_sentence=self.min_sentence, max_sentence=self.max_sentence)
        self.inp_vocab_size = len(self.inp_builder.word_index) + 1
        self.tar_vocab_size = len(self.tar_builder.word_index) + 1

        self.inp_tensor, self.tar_tensor,self.inp_builder,self.tar_builder = DataLoader(self.inp_lang_path,
                                                                                       self.tar_lang_path,
                                                                                       min_sentence=self.min_sentence,
                                                                                       max_sentence=self.max_sentence).tokenize()
        # Initialize optimizer
        if self.use_lr_schedule:
            self.optimizer = tf.keras.optimizers.Adam(CustomSchedule(self.embedding_size, self.learning_rate, self.warmup_steps))
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Inittalize Bleu score
        if self.use_bleu:
            self.bleu_score = Bleu_score()
        
        # Initialize encoder and decoder
        self.encoder = Encoder(self.inp_vocab_size, self.embedding_size, self.hidden_units)

        # Initialize decoder with mode attention

        if self.train_mode.lower() == "attention":
            if self.attention_mode.lower() == "luong":
                self.decoder = LuongDecoder(self.tar_vocab_size, self.embedding_size, self.hidden_units)
            else:
                self.decoder = BahdanauDecoder(self.tar_vocab_size, self.embedding_size, self.hidden_units)
        else:
            self.decoder = Decoder(self.tar_vocab_size, self.embedding_size, self.hidden_units)


        # Initialize Translation
        self.checkpoint_prefix  = os.path.join(self.path_save, "ckpt")
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder,
                                              optimizer=self.optimizer)
        
        if self.retrain:
            self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save))
            print("Restored from checkpoint:", tf.train.latest_checkpoint(self.path_save)).expect_partial()

        def train_step(self, inp, tar):
            loss = 0
            with tf.GradientTape() as tape:
                # teacher forcing
                # inp: (batch_size, seq_len_inp)
                # tar: (batch_size, seq_len_tar)
                self.tar_builder.word_index['<sos>'] = 1
                sos = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']]*self.BATCH_SIZE,),shape=(-1, 1))  # Start of sequence token
                
                encoder_outputs, state = self.encoder(inp)
                decoder_input = tf.concat([sos, tar[:, :-1]], axis=1)  # Concatenate <sos> token with the target sequence
                decoder_target = tar[:, 1:] # example: [2, 3, 4] -> [3, 4, 5] (shifted right)

                if self.train_mode.lower() == "attention":
                    decoder_output, _, _ = self.decoder(decoder_input, encoder_outputs, state)
                else:
                    decoder_output, _ = self.decoder(decoder_input, state)

                loss = MaskedSoftmaxCELoss()(decoder_target, decoder_output)

            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
            return loss