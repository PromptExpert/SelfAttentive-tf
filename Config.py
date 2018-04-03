class Config(object):
    def __init__(self,args,config_preprocess):
        self._embedding_size = args.embedding_size
        self._rnn_size_sent = args.rnn_size_sent
        self._sent_length = config_preprocess['sent_length']
        self._vocab_size = config_preprocess['vocab_size']
        self._num_labels = config_preprocess['num_labels']
        self._batch_size = args.batch_size
        self._attn_dim_sent = args.attn_dim_sent
        self._num_feature = args.num_feature
        self._use_gru = args.use_gru

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def rnn_size_sent(self):
        return self._rnn_size_sent

    @property
    def sent_length(self):
        return self._sent_length

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def attn_dim_sent(self):
        return self._attn_dim_sent

    @property
    def num_feature(self):
        return self._num_feature

    @property
    def use_gru(self):
        return self._use_gru
