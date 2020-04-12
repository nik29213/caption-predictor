from flask import Flask, render_template, redirect, request
import caption_script

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/', methods = ['POST'])
def submit_data():
	if request.method == 'POST':
		f = request.files['userfile']
		
		path="static/"+f.filename
		f.save(path)
		
		caption=caption_script.caption_this_img(path)
		ur_result={
		"image":path,
		"caption":caption
		}
	return render_template("index.html",your_result=ur_result)

if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)
