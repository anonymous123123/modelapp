from flask import Flask, make_response, request,jsonify
from model import GBModel
from io import StringIO
import pandas as pd
import pandas as pd
from parseHelper import parseJson


model = GBModel()
app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Get predictions</h1>
                <h3>Upload a csv file</h3>

                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """

@app.route('/transform', methods=["POST"])
def transform_view():
    request_file = request.files['data_file']

    if not request_file:
        return "No file"

    df = pd.read_csv(request_file,sep=";")
    predictions = model.getPredictions(df)

    s = StringIO()
    predictions.to_csv(s,index = None, header=True,sep=";")

    response = make_response(s.getvalue())

    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response



@app.route("/json", methods=["POST"])
def json_example():

    if request.is_json:

        request_data = parseJson(request.get_data(as_text=True))
        s = StringIO(request_data)
        df = pd.read_csv(s,sep=";")
        prediction = model.getSinglePrediction(df)

        res = make_response(str(round(prediction,4))+"\n", 200)
        return res

    else:

        return make_response(jsonify({"message": "Request body must be JSON"}), 400)

if __name__ == '__main__':
    app.run()


#curl localhost:5000/post -d '{"foo": "bar"}' -H 'Content-Type: application/json'
#curl --header "Content-Type: application/json" --request POST --data '{"name":"Julian","message":"Posting JSON data to Flask!"}' http://127.0.0.1:5000/json