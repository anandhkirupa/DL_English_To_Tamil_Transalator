class SimpleTokenizer:
    def __init__(self, sentences, min_freq=1):
        # Initialize dictionaries for word-to-index and index-to-word mappings
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_freq = {}

        # Count word frequencies across all sentences
        for sentence in sentences:
            for word in sentence.lower().split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Only include words that meet the minimum frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word

    def encode(self, sentence):
        """Converts a sentence into a list of token ids. Unknown words are mapped to <unk>."""
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence.lower().split()]

    def decode(self, indices):
        """Converts token ids back to a sentence."""
        return ' '.join(self.idx2word.get(idx, '<unk>') for idx in indices)

    def __len__(self):
        return len(self.word2idx)