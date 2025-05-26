import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hiden_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hiden_dim = hiden_dim
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = LSTM(   units = hiden_dim,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform')
        
    def call(self,x,*args, **kwargs):
        """
        Forward pass of the encoder.
        :param x: Input tensor of shape (batch_size, sequence_length)
        :return: Output tensor and hidden state
        """
        batch_size = tf.shape(x)[0]
        # Initialize hidden state and cell state
        first_hidden_state, first_cell_state = self.init_hidden_state(batch_size)
        x = self.embedding(x)
        output, h, c = self.lstm(x, initial_state=[first_hidden_state, first_cell_state])
        # output shape: (batch_size, sequence_length, hiden_dim)
        return output, [h, c]  # Return the last hidden state and cell state

    def init_hidden_state(self, batch_size):
        """
        Initialize the hidden state of the LSTM.
        :param batch_size: Size of the batch
        :return: Initial hidden state and cell state
        """
        return (tf.zeros((batch_size, self.hiden_dim)), # hidden state
                tf.zeros((batch_size, self.hiden_dim))) # cell state
        
if __name__ == "__main__":
    encoder = Encoder(vocab_size=10000, embedding_dim=256, hiden_dim=512)
    sample_input = tf.constant([[1, 2, 3], [4, 5, 6]]) # Example input tensor of shape (batch_size, sequence_length)
    sample_input = tf.cast(sample_input, dtype=tf.int32)  # Ensure input is of type int32
    output, states = encoder(sample_input)
    print("Output shape:", output.shape)
    print("Hidden state shape:", states[0].shape)
    print("Cell state shape:", states[1].shape)
    