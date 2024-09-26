from flask import Flask,jsonify,request,url_for
import model as md



app= Flask(__name__)
@app.route("/get_pred",methods=["POST"])
def get_pred(filename):
    prediction = md.model_pred(md.model_prob(filename))
    data={"Prediction" : prediction}
    return jsonify(data)

@app.route("/")
def hello_world():
    return "Hello World"


if __name__=="__main__":
    md.load_model()
    #url_for("hello_world")
    app.run()