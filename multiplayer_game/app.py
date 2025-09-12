from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # You can later pass random letters dynamically
    return render_template("index.html", letter="A", p1_score=0, p2_score=0)

if __name__ == "__main__":
    app.run(debug=True)
