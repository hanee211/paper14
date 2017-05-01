import numpy as np
#read embedding

word_file = 'test_word'
embedding_file = 'test_word_embedding_pad_eos.txt'
sentence_file = 'test_text.txt'
def preprocessing(text):
	_l = text.strip()
	_l = _l.replace('\n', '')
	#_l = _l.lower()
	
	return _l
	
def get_wordList_from_text():
	
	t_wordList = list()
	idx2word = list()
	
	with open('./data/' + sentence_file, 'r') as f:
		_l = preprocessing(f.read())
		_l = _l.split(' ')

		t_wordList.extend(_l)
	
	t_wordList = set(t_wordList)
	idx2word.append('PAD')
	idx2word.append('EOS')
	idx2word.extend(t_wordList)
	word2idx = {w:i for i , w in enumerate(idx2word)}
	return idx2word, word2idx
	
def get_wordList():
	
	t_wordList = list()
	idx2word = list()
	
	with open('./data/' + word_file, 'r') as f:
		_l = preprocessing(f.read())
		_l = _l.split(' ')

		t_wordList.extend(_l)
	
	t_wordList = set(t_wordList)
	idx2word.append('PAD')
	idx2word.append('EOS')
	idx2word.extend(t_wordList)
	word2idx = {w:i for i , w in enumerate(idx2word)}
	return idx2word, word2idx

def get_sentences():
	idx2word, word2idx = get_wordListFromFile()
	
	sentence_list = list()
	with open('./data/' + sentence_file, 'r') as f:
		for l in f:
			_l = preprocessing(l)
			_l = _l.split(' ')
		
			t = list()
			for w in _l:
				if w in word2idx:
					t.append(word2idx[w])
					
			if len(t) <= 50:
				sentence_list.append(t)

	return sentence_list
	
	
def get_wordEmbeddings():
	embedding = np.loadtxt('./data/' + embedding_file, dtype=np.float32)
	#embedding = np.loadtxt('./data/data_word_embedding.txt')
	return embedding

def get_wordListFromFile():
	idx2word = list()
	
	with open('./data/' + word_file, 'r') as f:
		for l in f:
			l = l.replace('\n', '')
			idx2word.append(l)
			
	word2idx = {w:i for i,w in enumerate(idx2word)}
		
	return idx2word, word2idx
	
	
def get_embeddingLookup():
	wordList = get_wordList()
	wordEmbeddings = get_wordEmbeddings()
	
	word2id = {w:i for i, w in enumerate(wordList)}	
	word2em = {word:embedding for word, embedding in zip(wordList, wordEmbeddings)}
	return word2em, word2id


if __name__ == '__main__':
	sentences = get_sentences()
	print(sentences[0])
	print(sentences[1])
	print(sentences[2])
	print(len(sentences))
	
