# from flask import Flask, render_template, flash, request, url_for
# import numpy as np
# import pandas as pd
# import re
# import os
# import tensorflow as tf 
# from numpy import array
# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras.models import load_model
# from keras.utils.data_utils import pad_sequences

# IMG_FOLDER = os.path.join('static', 'img_pool')

# app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = IMG_FOLDER

# def init():
#     global model
#     model = load_model('sentiment_analysis_model.h5')
#     # graph = tf.Graph()


# @app.route('/', methods = ['GET', 'POST'])
# def home():

#     return render_template("home.html")

# @app.route('/sentiment_analysis_prediction', methods = ['GET', 'POST'])
# def sent_any_prediction():
#     if request.method == 'POST':
#         text = request.form['text']
#         Sentiment = ''
#         max_review_length = 500
#         word_2_id = imdb.get_word_index()
#         strip_special_chars = re.compile("[^A-Za-z0-9]+")
#         text = text.lower().replace("<br />", " ")
#         text = re.sub(strip_special_chars, " ", text.lower())

#         words = text.split()
#         x_test = [[word_2_id[word] if (word in word_2_id and word_2_id[word]<=20000) else 0 for word in words]]
#         x_test = pad_sequences(x_test, maxlen=max_review_length)
#         vector = np.array([x_test.flatten()])

#         # with graph.as_default():
#         probability = model.predict(array([vector][0]))[0][0]
#         class1 = 1 if probability > 0.01 else 0

#         if class1 == 0:
#             Sentiment = "Negative"
#             img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
#         else:
#             Sentiment = "Positive"
#             img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.png')

#     return render_template('home.html', text = text, sentiment = Sentiment, probability = probability, image = img_filename)


# if __name__ == "__main__":
#     init()
#     app.run()



# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return render_template("index.html", title="Hello")




from flask import Flask, render_template, flash, request, url_for
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf 
import tensorflow_text as text
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils.data_utils import pad_sequences

IMG_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

def init():
    global model
    model = tf.saved_model.load('imdb_bert/')
    # graph = tf.Graph()



@app.route('/', methods = ['GET', 'POST'])
def home():

    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['GET', 'POST'])
def sent_any_prediction():
    if request.method == 'POST':
        text = request.form['text']
        print(text)
        get_text = [text]
        resulted_text = tf.sigmoid(model(tf.constant(get_text)))
        Sentiment = ''
    
        # result_for_printing = \
        #     [f'input: {get_text[i]:<30} : score: {resulted_text[i][0]:.6f}'
        #                             for i in range(len(get_text))]
  
        # prob = print_my_examples(get_text, resulted_text)
        res = tf.get_static_value(resulted_text[0][0], partial=False)
        class1 = 1 if res >= 0.5 else 0

        if class1 == 0:
            Sentiment = "Negative"
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        else:
            Sentiment = "Positive"
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.png')
        
        # serving_results = model \
        #     .signatures['serving_default'](tf.constant(get_text))

        # serving_results = tf.sigmoid(serving_results['classifier'])


    return render_template('home.html', text = text, sentiment = Sentiment, probability = res, image = img_filename)


if __name__ == "__main__":
    init()
    app.run()