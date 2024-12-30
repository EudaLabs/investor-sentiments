import wandb

wandb.login()
wandb.init(project='model-comparison', name='word2vec_vs_logreg')

import os
import joblib
from gensim.models import Word2Vec

BASE_DIR = '/Users/efecelik/Desktop/investor-sentiments'
word2vec_path = os.path.join(BASE_DIR, 'word2vec_model.model')
logreg_path = os.path.join(BASE_DIR, 'llogistic_model.pkl')

# Modelleri yükle
word2vec_model = Word2Vec.load(word2vec_path)
logreg_model = joblib.load(logreg_path)
