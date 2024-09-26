from flask import Flask,jsonify,request,url_for,render_template
import model as md



app= Flask(__name__)
@app.route("/get_pred",methods=["POST"])
def get_pred(filename):
    prediction = md.model_pred(md.model_prob(filename))
    data={"Prediction" : prediction}
    return jsonify(data)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/handle_form",methods=["POST"])
def handle_form():
    return "You did it"
    ufile=request.files['image']
    print(ufile.filename)
    if ufile.filename!='':
        return get_pred(ufile.filename)


if __name__=="__main__":
    md.load_model()
    #url_for("hello_world")
    app.run()