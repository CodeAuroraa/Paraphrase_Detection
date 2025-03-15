#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from similarity import compute_similarity
from model import detect_paraphrase

app = Flask(__name__)

@app.route('/paraphrase', methods=['POST'])
def paraphrase_api():
    """API endpoint to compare two texts and detect paraphrases."""
    data = request.json
    text1, text2 = data.get('text1'), data.get('text2')

    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400

    similarity_score = compute_similarity(text1, text2)
    result = detect_paraphrase(text1, text2)

    return jsonify({
        "cosine_similarity": similarity_score,
        "paraphrase": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires a specified port


