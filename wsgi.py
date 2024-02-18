from flask import Flask, render_template, redirect, url_for,request
from flask import make_response, send_from_directory
import process
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save(f.filename)
        result = process.main(f)
        result.save('static/'+result.filename)
        if result.filename.endswith("mp4"):
            return render_template("video.html", result='static/'+result.filename)
        else:
            return render_template("image.html", result='static/'+result.filename)

if __name__ == "__main__":
    app.run(debug = True)
