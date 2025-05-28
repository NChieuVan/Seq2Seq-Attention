import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM

class BahdanauAttention(Layer):
    def __init__(self, hidden_units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = Dense(hidden_units)
        self.W2 = Dense(hidden_units)
        self.V = Dense(1)

    def call(self, encoder_outputs, hidden_state):
        # encoder_outputs: (batch, seq_len, hidden_units)
        # hidden_state: (batch, hidden_units)
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)  # (batch, 1, hidden_units)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(hidden_with_time_axis)))  # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)
        context_vector = attention_weights * encoder_outputs  # (batch, seq_len, hidden_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_units)
        return context_vector, attention_weights

class BahdanauDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, **kwargs):
        super(BahdanauDecoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(hidden_units,
                         return_sequences=True,
                         return_state=True,
                         kernel_initializer="glorot_uniform")
        self.attention = BahdanauAttention(hidden_units)
        self.fc = Dense(vocab_size)

    def call(self, x, encoder_outputs, state):
        # x: (batch, 1) -- giả sử decode từng token 1 lần
        x = self.embedding(x)  # (batch, 1, embedding_dim)
        context_vector, attention_weights = self.attention(encoder_outputs, state[0])  # state[0]: hidden state
        context_vector = tf.expand_dims(context_vector, 1)  # (batch, 1, hidden_units)
        x = tf.concat([x, context_vector], axis=-1)  # (batch, 1, embedding_dim + hidden_units)
        output, state_h, state_c = self.lstm(x, initial_state=state)
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch, hidden_units)
        x = self.fc(output)  # (batch, vocab_size)
        return x, [state_h, state_c], attention_weights

# Ví dụ chạy
if __name__ == "__main__":
    batch_size = 2
    vocab_size = 10000
    embedding_dim = 256
    hidden_units = 512

    decoder = BahdanauDecoder(vocab_size, embedding_dim, hidden_units)

    sample_input = tf.constant([[1], [2]], dtype=tf.int32)  # batch=2, seq_len=1 (mỗi lần decode 1 token)
    encoder_outs = tf.random.normal((batch_size, 10, hidden_units))
    initial_state = [tf.zeros((batch_size, hidden_units)), tf.zeros((batch_size, hidden_units))]

    output, states, attn_weights = decoder(sample_input, encoder_outs, initial_state)

    print("Output shape:", output.shape)  # (batch_size, vocab_size)
    print("States shapes:", states[0].shape, states[1].shape)
    print("Attention weights shape:", attn_weights.shape)  # (batch_size, seq_len_encoder, 1)
#     print("Context vector shape:", states[0].shape)  # (batch_size, hidden_units)
#     print("Attention weights shape:", attn_weights.shape)  # (batch_size, seq_len_encoder, 1)