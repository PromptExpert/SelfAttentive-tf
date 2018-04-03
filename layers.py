'''
sub-components of the STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
'''
import tensorflow as tf

def get_lengths(batch):
    '''
    compute the actual lengths for each of the sequences in the batch
    the input shape is (batch, max_length_snet)
    return shape is (batch,)
    '''
    batch = tf.sign(batch)
    lengths = tf.reduce_sum(batch,1)
    return lengths

class Embedding(object):
    def __init__(self,vocabulary_size,embedding_size):
        self.embeddings_matrix = tf.get_variable('embeddings_matrix', [vocabulary_size, embedding_size],dtype = tf.float32) #创建一个词嵌入矩阵

    def __call__(self,sents):
        return tf.nn.embedding_lookup(self.embeddings_matrix,sents)


class SentEncoder(object):
    def __init__(self,rnn_size_sent,sent_length,batch_size,embedding_size,use_gru = False):
        self.rnn_size_sent = rnn_size_sent
        self.sent_length = sent_length
        self.embedding_size = embedding_size
        self.rnn_cell_fw = tf.contrib.rnn.GRUCell(self.rnn_size_sent) if use_gru else tf.contrib.rnn.LSTMCell(self.rnn_size_sent)
        self.rnn_cell_bw = tf.contrib.rnn.GRUCell(self.rnn_size_sent) if use_gru else tf.contrib.rnn.LSTMCell(self.rnn_size_sent)


    def __call__(self,sents_embeddings,lengths):
        ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell_fw,
                                                                             cell_bw=self.rnn_cell_bw,
                                                                             inputs=sents_embeddings,
                                                                             dtype=tf.float32,
                                                                             scope = "sentence_encoder",
                                                                             sequence_length = lengths)
        H = tf.concat((fw_outputs, bw_outputs), 2)
        return H


class SentAttention(object):
    '''
    A = softmax(Ws2(tanh(Ws1 H^T))
    M = AH
    '''
    def __init__(self,rnn_size_sent,sent_length,attn_dim_sent,num_feature):
        self.rnn_size_sent = rnn_size_sent
        self.sent_length = sent_length
        self.attn_dim_sent = attn_dim_sent
        self.num_feature = num_feature

    def __call__(self,H):
        result1 = tf.contrib.layers.fully_connected(H, self.attn_dim_sent, activation_fn=tf.nn.tanh,scope='Ws1') #tanh(Ws1 H^T),(batch_size,sent_lengt,d_a)
        result2 = tf.contrib.layers.fully_connected(result1, self.num_feature, activation_fn=None,scope='Ws2') #(batch_size,sent_lengt,r)
        A = tf.nn.softmax(result2, axis=1)
        M = tf.matmul(H, A,transpose_a = True)
        return A,M

class MatrixEncoder(object):
    '''
    transform the sentence embeding - a matrix into a sequence of annotations
    '''
    def __init__(self,rnn_size_matrix,num_feature,rnn_size_sent,use_gru = False):
        self.rnn_size_matrix = rnn_size_matrix
        self.sequence_length = num_feature
        self.element_size = rnn_size_sent*2
        self.rnn_cell_fw = tf.contrib.rnn.GRUCell(self.rnn_size_matrix) if use_gru else tf.contrib.rnn.LSTMCell(self.rnn_size_matrix)
        self.rnn_cell_bw = tf.contrib.rnn.GRUCell(self.rnn_size_matrix) if use_gru else tf.contrib.rnn.LSTMCell(self.rnn_size_matrix)


    def __call__(self,sentence_matrix):
        ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell_fw,
                                                                             cell_bw=self.rnn_cell_bw,
                                                                             inputs=sentence_matrix,
                                                                             dtype=tf.float32,
                                                                             scope = "matrix_encoder",
                                                                             )
        annotations = tf.concat((fw_outputs, bw_outputs), 2)
        return annotations


class MatrixAttention(object):
    '''
    Transform the annotations into a vector
    '''
    def __init__(self,num_feature,attn_dim_matrix):
        self.sequence_length = num_feature
        self.attn_dim_matrix = attn_dim_matrix

    def __call__(self,annotations):
        result1 = tf.contrib.layers.fully_connected(annotations, self.attn_dim_matrix, activation_fn=tf.nn.tanh,scope='Ws') #tanh(Ws H^T),(batch_size,r,d_a)
        result2 = tf.contrib.layers.fully_connected(result1, 1 , activation_fn=None,scope='ws2') #(batch_size,r,1)
        a = tf.nn.softmax(result2, axis=1)
        vector = tf.matmul(annotations, a ,transpose_a = True) #矩阵跟向量的乘法就是矩阵列的线性组合（加权和），(batch_size,rnn_size_matrix,1）
        return tf.squeeze(vector)
