from webapp import app
import nltk

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words',])

# let Heroku pick a port and IP
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)