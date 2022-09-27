# Import libaries
import csv
import pandas as pd
from flask import Flask, render_template, request
from model import run_model
import os

# cwd = os.getcwd()

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/getdata", methods=["POST", "GET"])
def getdata():
    if request.method == "POST":
        upload_file = request.form["upload_file"]
        data = []
        with open(upload_file, encoding='UTF8') as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
            data = pd.DataFrame(data)
        result = run_model(upload_file)
        return render_template("index.html", data=data.to_html(border=True, header=False, index=False, classes='table table-stripped'), result=result)


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
