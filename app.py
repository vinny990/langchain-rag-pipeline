from flask import Flask, request, jsonify
import rag_chain

app = Flask(__name__)

CHAT_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HR Policy Assistant</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; }
    h1 { font-size: 1.4rem; }
    #log { border: 1px solid #ddd; border-radius: 6px; padding: 12px; min-height: 200px;
           max-height: 500px; overflow-y: auto; margin-bottom: 12px; background: #fafafa; }
    .q { color: #333; margin: 8px 0 2px; }
    .a { color: #1a6b3c; margin: 2px 0 8px; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 1rem; }
    button { padding: 8px 16px; background: #1a6b3c; color: #fff; border: none;
             border-radius: 4px; cursor: pointer; font-size: 1rem; }
    button:disabled { opacity: 0.5; }
  </style>
</head>
<body>
  <h1>HR Policy Assistant</h1>
  <div id="log"></div>
  <form id="form">
    <input id="q" type="text" placeholder="Ask an HR policy question..." autocomplete="off" required>
    <button type="submit" id="btn">Ask</button>
  </form>
  <script>
    const log = document.getElementById("log");
    const form = document.getElementById("form");
    const input = document.getElementById("q");
    const btn = document.getElementById("btn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const question = input.value.trim();
      if (!question) return;

      log.innerHTML += `<p class="q"><strong>You:</strong> ${question}</p>`;
      input.value = "";
      btn.disabled = true;

      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      log.innerHTML += `<p class="a"><strong>Assistant:</strong> ${data.answer || data.error}</p>`;
      log.scrollTop = log.scrollHeight;
      btn.disabled = false;
      input.focus();
    });
  </script>
</body>
</html>"""


@app.route("/")
def index():
    return CHAT_UI


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    answer = rag_chain.ask(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
