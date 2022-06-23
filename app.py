from flask import Flask,jsonify, request, render_template
import numpy as np
import joblib
import keras
import librosa

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = 'static/Emotion_Voice_Detection_Model66.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions=self.loaded_model.predict(x) 
        classes_x=np.argmax(predictions,axis=1)
        a=self.convert_class_to_emotion(classes_x)
        return a

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral 😐',
                            '1': 'calm 🙂',
                            '2': 'happy 😁',
                            '3': 'sad 😟',
                            '4': 'angry 😡',
                            '5': 'fearful 😨',
                            '6': 'disgust 🤢',
                            '7': 'surprised 😮'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":

        file = request.form['todo']
        live_prediction = LivePredictions(file='static/Audios/'+file)
        b=live_prediction.make_predictions()
        return jsonify({'output':'You are ' + b + ', right?'})
    else:
        b = ""
        
    return render_template("index.html")

# Running the app
if __name__ == '__main__':
    app.run(debug = True)