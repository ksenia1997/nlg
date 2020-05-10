file_data = open('SST_data_extraction/datasetSentences.txt', "r")
file_sentiment = open('SST_data_extraction/sentiment_labels.txt', "r")
counter = 0

negative_sen = open('negative_sentences.txt', "w")
positive_sen = open('positive_sentences.txt', "w")
for sentence in file_data:
	counter += 1
	sentiment_f = file_sentiment.readline()
	if counter != 1:
		new_sentence = ' '.join(sentence.split()[1:])
		sentiment = float(sentiment_f.split('|')[1][:-1])
		if sentiment <= 0.4:
			negative_sen.write(new_sentence + "\n")
			negative_sen.write("\n")
		if sentiment > 0.6:
			positive_sen.write(new_sentence + "\n")
			positive_sen.write("\n")

