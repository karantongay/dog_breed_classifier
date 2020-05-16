import json

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from predictor import algorithm
import numpy as np
# from werkzeug import secure_filename


app = Flask(__name__, static_url_path='/static')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

		
# web page that handles user query and displays model results
@app.route('/go', methods = ['GET', 'POST'])
def go():
    if request.method == 'POST':
      f = request.files['file']
      f.save('file.jpg')


    # use model to predict classification for query
    breed = algorithm('file.jpg').partition('.')[-1]

    print(app.instance_path)
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html', breed = breed
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()