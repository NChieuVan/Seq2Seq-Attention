import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM

class LuongAttention(Layer):
    def __init__(self, hidden_units, **kwargs):
        super(LuongAttention, self).__init__(**kwargs)
        self.Wa = Dense(hidden_units)

    def call(self, encoder_outs, decoder_outs, *args, **kwargs):
        """
        encoder_outs: (batch, seq_len_enc, hidden_units)
        decoder_outs: (batch, seq_len_dec, hidden_units)
        """
        score = tf.matmul(decoder_outs, self.Wa(encoder_outs), transpose_b=True)  # (batch, seq_len_dec, seq_len_enc)
        alignment = tf.nn.softmax(score, axis=2)  # attention weights
        context_vector = tf.matmul(alignment, encoder_outs)  # (batch, seq_len_dec, hidden_units)
        return context_vector, alignment

class LuongDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        super(LuongDecoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.attention = LuongAttention(hidden_units=hidden_units)
        self.dense = Dense(vocab_size)

    def call(self, x, encoder_outs, state, *args, **kwargs):
        """
        x: (batch, seq_len_dec)
        encoder_outs: (batch, seq_len_enc, hidden_units)
        state: [h, c], each (batch, hidden_units)
        """
        # x = tf.expand_dims(x, axis=1)  # process 1 timestep input at a time
        x = self.embedding(x)  # (batch, 1, embedding_size)
        decode_outs, state_h, state_c = self.decode_layer_1(x, initial_state=state)
        context_vector, att_weights = self.attention(encoder_outs, decode_outs)
        concat = tf.concat([decode_outs, context_vector], axis=-1)  # (batch, 1, hidden_units*2)
        concat = tf.reshape(concat, (-1, concat.shape[2]))  # (batch, hidden_units*2)
        outs = self.dense(concat)  # (batch, vocab_size)
        return outs, [state_h, state_c], att_weights
        
if __name__ == "__main__":
    decoder = LuongDecoder(vocab_size=10000, embedding_size=256, hidden_units=512)
    sample_input = tf.constant([[1, 2, 3], [4, 5, 6]])  # Example input tensor of shape (batch_size, sequence_length)
    sample_input = tf.cast(sample_input, dtype=tf.int32)  # Ensure input is of type int32
    encoder_outs = tf.random.normal((2, 10, 512))  # Example encoder outputs
    initial_state = [tf.zeros((2, 512)), tf.zeros((2, 512))]  # Example initial state
    output, states, att_weights = decoder(sample_input, encoder_outs, initial_state)
    print("Output shape:", output.shape)
    print("Hidden state shape:", states[0].shape)
    print("Cell state shape:", states[1].shape)
    print("Attention weights shape:", att_weights.shape)    