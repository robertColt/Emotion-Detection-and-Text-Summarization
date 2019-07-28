import eel
import text_summarization as summarizer
import emotion_detection

eel.init("web")


@eel.expose
def summarize(text, compression=0.5):
    print("\n\nSummarizing text")
    sentences, stemmed_sentences, clean_sentences = summarizer.get_sentences(text)
    # similarity_matrix = summarizer.similarity_matrix_common_tokens(stemmed_sentences)
    print("Embeddings:")
    summary_embedding = summarizer.create_summary(sentences, summarizer.similarity_matrix_embeddings(clean_sentences), compression=compression)
    # summary_embedding = ""
    print("\n\nTokens")
    summary_tokens = summarizer.create_summary(sentences, summarizer.similarity_matrix_common_tokens(stemmed_sentences), compression=compression)
    print("\n\n")
    # summary = summarizer.create_summary(sentences, similarity_matrix, compression)
    print("Summary ready...sending")
    eel.on_summary_ready(summary_embedding, summary_tokens)


@eel.expose
def detect_emotions(text):
    # translated = emotion_detection.translate(text)
    # lemma_pos = emotion_detection.get_lemmatized_words_en(translated)
    # (emotion_str, emotion_scores_str) = emotion_detection.calculate_emotion_scores_depeche(lemma_pos)

    sentences, stemmed_sentences, clean_sentences = summarizer.get_sentences(text)
    (emotion_str, emotion_scores_str) = emotion_detection.calculate_emotion_scores_roemolex(sum(clean_sentences, []))

    eel.on_emotions_ready(emotion_scores_str, emotion_str)


web_app_options = {
    'mode': "chrome-app", #or "chrome"
    'port': 8081,
    'chromeFlags': []
}


eel.start("main_.html", options=web_app_options)

while True:
    eel.sleep(10)
