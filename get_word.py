import numpy as np
#read embedding

word_file = 'text1.txt'

def get_wordList():
	t_wordList = list()
	with open('./data/' + word_file, 'r') as f:
		_l = f.read().strip()
		_l = _l.replace('\n', '')
		_l = _l.split(' ')

		t_wordList.extend(_l)
	
	wordList = set(t_wordList)
	
	return wordList

	
def get_wordEmbeddings():
	embedding = np.loadtxt('./data/' + embedding_file)
	#embedding = np.loadtxt('./data/data_word_embedding.txt')
	return embedding


def get_embeddingLookup():
	wordList = get_wordList()
	wordEmbeddings = get_wordEmbeddings()
	
	word2id = {w:i for i, w in enumerate(wordList)}	
	word2em = {word:embedding for word, embedding in zip(wordList, wordEmbeddings)}
	return word2em, word2id


if __name__ == '__main__':
	words = get_wordList()
	#print(words)