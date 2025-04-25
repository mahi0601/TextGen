from flask import Flask, request, jsonify
from app.interface import predict, predict_and_evaluate

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ Text Generator API is running! Use POST /generate</h2>"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")
    reference = data.get("reference")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if reference:
        result = predict_and_evaluate(prompt, reference)
    else:
        result = {"generated": predict(prompt)}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
