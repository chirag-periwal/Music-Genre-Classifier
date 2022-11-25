from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import pickle
import librosa
import numpy as np
import os
from xgboost import XGBClassifier



classifier=pickle.load(open('classifier.pkl','rb'))
xgb=pickle.load(open('xgb.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      # f.save(os.path.join(app.config['UPLOAD_FOLDER'], f))
   songname=f.filename
   def predict(li):
      results = {1: 'Blues', 2: 'Classical', 3: 'Country', 4: 'Disco', 5: 'HipHop', 6: 'Jazz', 7: 'Metal', 8: 'Pop', 9: 'Reggae', 10: 'Rock'}
      number=xgb.predict([li])[0]
      return results[number+1]
   def feature_extract(songname):
      y, sr = librosa.load(songname, mono=True, duration=30)
      chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
      rmse = librosa.feature.rms(y=y)
      spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
      spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
      rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
      zcr = librosa.feature.zero_crossing_rate(y)
      mfcc = librosa.feature.mfcc(y=y, sr=sr)
      to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
      for e in mfcc:
         to_append += f' {np.mean(e)}'
      # to_append
      li = list(to_append.split(" "))
      li=[float(i) for i in li]
      return li
   predicted=predict(feature_extract(songname))
   return render_template('genre.html',data=predicted)
   # return predicted

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000,debug=True)