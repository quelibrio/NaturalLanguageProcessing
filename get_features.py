import pandas as pd
import gensim
import W2VProcessing as processing
import pytextvec as pytextvec

#pytextvec.Tokenizer(input_f = 'data/receipt_lines.csv',
#                      output_f = 'data/receipt_tokenized.csv', 
#                      n_threads= 2,
#                      batch_size = 100)

pytextvec.Tokenizer('data/receipt_lines.csv','data/receipt_tokenized.csv')
processing.BigramsTrigrams('data/receipt_tokenized.csv', 'data/receipt_bigrams.csv', 'data/receipt_trigrams.csv')
processing.Word2Vec('data/receipt_tokenized.csv')
#tokenize.AI2.Tokenizer('tokenized.csv')