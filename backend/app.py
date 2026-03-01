from flask import Flask, jsonify
from flask_cors import CORS
from backend.simulator.process import Process

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "AI Smart CPU Scheduler Backend Running"})


@app.route("/test")
def test():
    p = Process(1, 0, 10, 1)
    return jsonify({"pid": p.pid})


if __name__ == "__main__":
    app.run(debug=True)