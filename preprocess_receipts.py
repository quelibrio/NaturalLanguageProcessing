import pandas as pd
import gensim
import W2VProcessing as processing
import pytextvec as pytextvec
import glob

def text_is_item (row):
   if float(row['price']) !=0 and float(row['cost']) != 0:
      return '1'
   return '0'

def text_category(row):
    if row['line']  < 900: return 1
    if row['line'] == 900: return 900
    if row['line'] == 941: return 941
    if row['line'] == 950: return 950
    if row['line'] == 980: return 980
    return 1000

all_files = glob.glob('data/SamplePrintScans/*_D_.csv')

df_concat = pd.DataFrame(columns= ['descript'])
df_labels = pd.DataFrame(columns= ['is_item'])

count = 0
for file in all_files:
    count+=1
    if count > 20:
        break
    
    df = pd.read_csv(file)
    df['descript'] = df['descript'].str.replace(r'[^A-Za-z0-9\w]+',' ')
    df['descript'] = df['descript'].str.lower()

    frames = [df_concat, df]
    print(df['descript'].head)
    df_concat = pd.concat(frames)

    df_training = pd.DataFrame(columns= ['is_item'])
    df_training['is_item'] = df.apply (lambda row: text_category(row), axis=1)
   
    frames = [df_labels, df_training]
    df_labels = pd.concat(frames)


df_concat['descript'].to_csv("data/receipt_lines.csv", encoding='utf-8', index=False)
df_labels['is_item'].to_csv("data/receipt_labels.csv", encoding='utf-8', index=False)