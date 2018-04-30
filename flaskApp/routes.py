from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__)

# Index page
@app.route('/')
def index():
	# Determine the selected feature
	return render_template("index.html")

if __name__ == '__main__':
	app.run(debug=True)