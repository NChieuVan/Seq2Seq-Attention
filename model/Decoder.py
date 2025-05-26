import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = LSTM(units=hidden_dim,
                         return_sequences=True,
                         return_state=True,
                         recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)  # Fully connected layer to output vocabulary size

    def call(self, x, initial_state=None):
        """
        Forward pass of the decoder.
        :param x: Input tensor of shape (batch_size, sequence_length)
        :param initial_state: Initial hidden and cell states
        :return: Output tensor and hidden state
        : Output LSTM shape: (batch_size, sequence_length, hidden_units)
        : state_h = hidden state of the LSTM shape: (batch_size, hidden_dim)
        : state_c = cell state of the LSTM shape: (batch_size, hidden_dim)
        """
        x = self.embedding(x)
        output, h, c = self.lstm(x, initial_state=initial_state)
        output = self.fc(output)  # Apply the fully connected layer
        return output, [h, c]
    
if __name__ == "__main__":
    decoder = Decoder(vocab_size=10000, embedding_dim=256, hidden_dim=512)
    sample_input = tf.constant([[1, 2, 3], [4, 5, 6]])  # Example input tensor of shape (batch_size, sequence_length)
    sample_input = tf.cast(sample_input, dtype=tf.int32)  # Ensure input is of type int32
    initial_state = [tf.zeros((2, 512)), tf.zeros((2, 512))]  # Example initial state
    output, states = decoder(sample_input, initial_state=initial_state)
    print("Output shape:", output.shape)
    print("Hidden state shape:", states[0].shape)
    print("Cell state shape:", states[1].shape)
    