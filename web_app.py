
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Import the reusable RAG analysis function.
from rag_engine import analyze_message


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please enter a message to analyze."
        }), 400

    if len(message) > 5000:
        return jsonify({
            "error": "Message must be 5,000 characters or fewer."
        }), 400

    try:
        result = analyze_message(message)
        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("RAG analysis failed")
        return jsonify({
            "error": "Analysis failed. Check the VS Code terminal for details."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
