import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import RomanianStemmer

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx
import string
import pickle

nltk.download('punkt')
nltk.download('stopwords')

raw_text = "Facebook plăteşte bani utilizatorilor care aleg să instaleze o aplicaţie care le monitorizează toată activitatea de pe telefon. Facebook a reluat campania în care utilizatorii cu vârste începând de la 18 ani pot alege să devină cobai pentru administratorii reţelei de socializare, instalând în schimbul unei recompense de până la 20 dolari pe lună o aplicaţie numită Study from Facebook. Dezvoltată în versiune pentru iOS şi Android, aplicaţia Study from Facebook vine înzestrată cu certificate de acces ROOT pe telefoanele iPhone şi poate urmări toate aspectele care ţin de folosirea telefonului mobil, activitatea web şi conversaţiile private, informaţiile ajungând direct pe serverele Facebook. Datele obţinute ajută Facebook să identifice din timp pericole care pot ameninţa supremaţia reţelei de soclializare, dezvoltând apoi strategii pentru blocarea rivalilor nou apăruţi, fie cumpărând pe bani mulţi businessul respectiv, sau copiind cu neruşinare noutăţile aduse de aceştia. Deşi este ceva mai transparentă faţă de utilizatorii care aleg să devină voluntari în schimbul unor sume de bani, aplicaţia Study from Facebook forţează la limită legislaţia care trasează drepturile utilizatorilor la propria intimitate. Mai exact, este vorba de folosirea certificatelor de tip Root Acces, incluse în mod uzual cu aplicaţii folosite de clienţi business pentru accesarea telefoanelor de serviciu ale angajaţilor. Însă aplicaţia Study from Facebook este instalată pe telefoanele personale ale utilizatorilor, urmărind aspecte ale vieţii personale în interesul unei companii private. Facebook dă asigurări că nu va colecta nume de utilizator şi parole pentru alte conturi deţinute de persoanele recrutate în programul de monitorizare. Voluntarii pentru aplicaţia Study from Facebook sunt atraşi prin reclame distribuite pe platformele de socializare proprii, folosind o companie numită Applause. Interesant este că participanţii sunt informaţi că participă la un parteneriat cu Facebook, punând accentul cât mai mult pe partea de recompensă bănească şi mai puţin pe aspectele care ţin de colectarea agresivă de informaţii. Datele colectate în cadrul studiului nu vor fi revândute sau predate companiilor de publicitate partenere, scopul declarat fiind cel de îmbunătăţire a serviciilor Facebook. Acceptarea ofertei adresată utilizatorilor Facebook din Statele Unite şi India presupune crearea şi a unui cont PayPal, în care să fie virată recompensa financiară. Vârsta declarată este verificată manual de angajaţii Facebook, comparând toate conturile de utilizator descoperite online. Nu este impusă semnarea unui acord de confidenţialitate, recruţii fiind astfel încurajaţi să atragă noi doritori."

word_embeddings = {}
lemmas = {}

with open("data/word_embeddings.dict", "rb") as inputfile:
    word_embeddings = pickle.load(inputfile)

with open("data/lemmas_ro.dict", "rb") as inputfile:
    lemmas = pickle.load(inputfile)

print(lemmas["abacele"])

def get_sentences(text):
    # tokenizing sentences
    sentences = sent_tokenize(text)
    print("Raw", sentences)

    # removing punctuation
    clean_sentences = [s.translate(str.maketrans('', '', string.punctuation)).lower().split(" ") for s in sentences]
    print("Removing punctuation", clean_sentences)

    # stemming the sentences and removing stop words
    stemmer = RomanianStemmer()
    stop_words = set(stopwords.words("romanian"))
    stemmed_sentences = []
    for sentence in clean_sentences:
        # split_sentence = sentence.split(" ")
        stemmed_sentences.append([stemmer.stem(word)
                                  for word in sentence
                                  if word not in stop_words])

    print("After stemming", stemmed_sentences)
    return sentences, stemmed_sentences, clean_sentences


def sentence_token_similarity(sentence1, sentence2):
    # get all words from both sentences
    words = list(set(sentence1 + sentence2))
    num_words = len(words)

    # create vectors for each sentence to count word occurence
    vector1 = [0] * num_words
    vector2 = [0] * num_words

    # count word occurences
    for index, word in enumerate(words):
        if word in sentence1:
            vector1[index] += 1
        if word in sentence2:
            vector2[index] += 1

    return 1 - cosine_distance(vector1, vector2)


def similarity_matrix_common_tokens(sentences):
    num_sentences = len(sentences)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                similarity_matrix[i][j] = sentence_token_similarity(sentences[i], sentences[j])
    return similarity_matrix


def common_tokens(sentence1, sentence2, verbose):
    from math import log
    words = list(set(sentence1 + sentence2))
    if verbose : print("\n\nCommon Words", words, "\n\n")
    similarity = 0
    for word in words:
        if word in sentence2 and word in sentence1:
            if verbose:
                print(word)
            similarity += 1

    return similarity / (log(len(sentence2)) + log(len(sentence1)))


def similarity_matrix_common_tokens2(sentences):
    num_sentences = len(sentences)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                similarity_matrix[i][j] = common_tokens(sentences[i], sentences[j], verbose=False)
    return similarity_matrix


def lemmatize_sentences(sentences):
    global lemmas
    lemmatized_sentences = []
    for sentence in sentences:
        lemmatized_sentences.append([lemmas.get(word, word) for word in sentence])
    print("lemmatized", lemmatized_sentences)
    return lemmatized_sentences


def sentence_embeddings(sentences):
    global word_embeddings
    sentence_vectors = []
    lemmatized_sentences = lemmatize_sentences(sentences)
    for sentence in lemmatized_sentences:
        if len(sentence) != 0:
            v = sum([word_embeddings.get(word, np.zeros((100,))) for word in sentence])
            v /= len(sentence) + 0.001
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    return sentence_vectors


def similarity_matrix_embeddings(sentences):
    num_sentences = len(sentences)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    sentence_vectors = sentence_embeddings(sentences)

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                similarity_matrix[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0][0]
    return similarity_matrix


def create_summary(original_sentences, similarity_matrix, compression=0.5):
    # build a graph from similarity matrix and compute the scores with pageRank algorithm

    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    print("\nScores", scores)
    similarity_graph.clear()

    with open("data/scores_common.csv", "wt") as score_fiile:
        for i in range(len(original_sentences)):
            score_fiile.write("{}\n".format(scores[i]))

    ranked_sentences = sorted([(scores[i], sentence, i) for i, sentence in enumerate(original_sentences)], reverse=True)

    print("\nranked sentences", ranked_sentences)
    num_sentences = similarity_matrix.shape[0]

    # compute the number of sentences in the final summary
    num_target_sentences = int(num_sentences * compression)
    print("choosing", num_target_sentences, "sentences", compression, "compression")
    final_summary = ""
    top_sentences = {}
    sentence_indices = []

    for i in range(num_target_sentences):
        sent_idx = ranked_sentences[i][2]
        top_sentences[sent_idx] = ranked_sentences[i][1]
        sentence_indices.append(sent_idx)

    sentidx1 = [x+1 for x in sentence_indices]
    print("\nTOP SENTENCES", sentidx1, sorted(sentidx1), "\n", top_sentences)
    while len(sentence_indices) > 0:
        min_sent_idx = min(sentence_indices)
        final_summary += "{}. {}\n".format(min_sent_idx+1, top_sentences[min_sent_idx])
        sentence_indices.remove(min_sent_idx)

    return final_summary


if __name__ == "__main__":

    sentences, stemmed_sentences, clean_sentences = get_sentences(raw_text)
    print("Sentence count: ", len(sentences))

    similarity_common = similarity_matrix_common_tokens2(lemmatize_sentences(clean_sentences))
    similarity_embeddings = similarity_matrix_embeddings(lemmatize_sentences(clean_sentences))
    print("\nFinal Summary\n", create_summary(sentences, similarity_common))

    np.savetxt("data/similarity_common_tokens.csv", similarity_common, delimiter=",", fmt="%.3f")
    # np.savetxt("data/similarity_embeddings.csv", similarity_embeddings, delimiter=",", fmt="%.3f")

