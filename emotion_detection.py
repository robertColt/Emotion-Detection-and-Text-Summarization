#!/usr/bin/python
# -*- coding: UTF-8 -*-
from py_translator import Translator
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
import pickle
from nltk.corpus import wordnet
import text_summarization
import sqlite3

nltk.download('averaged_perceptron_tagger')

emotion_lexicon = {}
emotion_list = []
roemolex = pd.read_csv("data/roemolex.csv")

cnx = sqlite3.connect('data/language.db')

lemma_dict = pd.read_sql_query("SELECT * FROM Lemmas", cnx)

with open("data/emotion_lexicon.dict", "rb") as infile:
    emotion_lexicon, emotion_list = pickle.load(infile)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_lemmatized_words_en(text):
    # tokenizing sentences
    lower_text = text.lower()
    word_tokens = word_tokenize(lower_text)
    print("Raw", word_tokens)
    pos_tags = pos_tag(word_tokens)
    print("POS tags", pos_tags)

    lemmatizer = WordNetLemmatizer()
    lemma_pos = []
    lemmas = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:
            lemma = lemmatizer.lemmatize(word, wordnet_pos)
            lemma_pos.append("{}#{}".format(lemma, wordnet_pos))
            lemmas.append(lemma)

    print("Lemmatized", lemma_pos, lemmas)
    return lemma_pos

    # removing punctuation
    # clean_sentences = [s.translate(str.maketrans('', '', string.punctuation)).lower().split(" ") for s in sentences]
    # print("Removing punctuation", clean_sentences)


def translate(text):
    return Translator().translate(text, dest="en").text


def calculate_emotion_scores_depeche(lemmatized_words):
    global emotion_list, emotion_lexicon
    cnt_emotions = len(emotion_list)
    score = np.zeros((cnt_emotions,))
    zero_array = np.zeros((cnt_emotions,))
    word_count = 0
    for word in lemmatized_words:
        word_score = emotion_lexicon.get(word, zero_array)
        score += word_score
        if not np.array_equal(zero_array, word_score):
            word_count += 1
            print("Found words {}/{}".format(word_count, len(lemmatized_words)))

    return display_emotion_scores(score, emotion_list)


def display_emotion_scores(score, emotions_list):
    for i, emotion in enumerate(emotions_list):
        print(emotion, score[i])

    emotion_str = " ".join(emotions_list)
    emotion_scores_str = " ".join(np.char.mod('%.2f', score))

    print(emotion_str, "\n", emotion_scores_str)
    return emotion_str, emotion_scores_str


def calculate_emotion_scores_roemolex(words):
    global roemolex, lemma_dict
    emotions = roemolex.columns[1:]
    emotion_count = len(emotions)
    overall_score = np.zeros((emotion_count,))

    found_words = 0
    for word in words:
        lemmas = lemma_dict[lemma_dict["nice_word"] == word].values
        found_word_cnt = len(lemmas)
        if found_word_cnt > 0:
            for i in range(found_word_cnt):
                lemma = lemmas[i, 1]
                emotion_scores = roemolex[roemolex["word"] == lemma].values
                if len(emotion_scores) > 0:
                    found_words += 1
                    print("found word {} {}/{}".format(lemma, found_words, len(words)))
                    overall_score += np.asarray(emotion_scores[0, 1:], dtype="int32")
                    break

    return display_emotion_scores(overall_score, emotions)


if __name__ == "__main__":

    ro_text = 'Facebook plăteşte bani utilizatorilor care aleg să instaleze o aplicaţie care le monitorizează toată activitatea de pe telefon. Facebook a reluat campania în care utilizatorii cu vârste începând de la 18 ani pot alege să devină cobai pentru administratorii reţelei de socializare, instalând în schimbul unei recompense de până la 20 dolari pe lună o aplicaţie numită Study from Facebook. Dezvoltată în versiune pentru iOS şi Android, aplicaţia Study from Facebook vine înzestrată cu certificate de acces ROOT pe telefoanele iPhone şi poate urmări toate aspectele care ţin de folosirea telefonului mobil, activitatea web şi conversaţiile private, informaţiile ajungând direct pe serverele Facebook. Datele obţinute ajută Facebook să identifice din timp pericole care pot ameninţa supremaţia reţelei de soclializare, dezvoltând apoi strategii pentru blocarea rivalilor nou apăruţi, fie cumpărând pe bani mulţi businessul respectiv, sau copiind cu neruşinare noutăţile aduse de aceştia. Deşi este ceva mai transparentă faţă de utilizatorii care aleg să devină voluntari în schimbul unor sume de bani, aplicaţia Study from Facebook forţează la limită legislaţia care trasează drepturile utilizatorilor la propria intimitate. Mai exact, este vorba de folosirea certificatelor de tip Root Acces, incluse în mod uzual cu aplicaţii folosite de clienţi business pentru accesarea telefoanelor de serviciu ale angajaţilor. Însă aplicaţia Study from Facebook este instalată pe telefoanele personale ale utilizatorilor, urmărind aspecte ale vieţii personale în interesul unei companii private. Facebook dă asigurări că nu va colecta nume de utilizator şi parole pentru alte conturi deţinute de persoanele recrutate în programul de monitorizare. Voluntarii pentru aplicaţia Study from Facebook sunt atraşi prin reclame distribuite pe platformele de socializare proprii, folosind o companie numită Applause. Interesant este că participanţii sunt informaţi că participă la un parteneriat cu Facebook, punând accentul cât mai mult pe partea de recompensă bănească şi mai puţin pe aspectele care ţin de colectarea agresivă de informaţii. Datele colectate în cadrul studiului nu vor fi revândute sau predate companiilor de publicitate partenere, scopul declarat fiind cel de îmbunătăţire a serviciilor Facebook. Acceptarea ofertei adresată utilizatorilor Facebook din Statele Unite şi India presupune crearea şi a unui cont PayPal, în care să fie virată recompensa financiară. Vârsta declarată este verificată manual de angajaţii Facebook, comparând toate conturile de utilizator descoperite online. Nu este impusă semnarea unui acord de confidenţialitate, recruţii fiind astfel încurajaţi să atragă noi doritori.'
    sentences, stemmed_sentences, clean_sentences = text_summarization.get_sentences(ro_text)
    # lemmatized_sent = text_summarization.lemmatize_sentences(clean_sentences)
    # lemmatized_words = sum(lemmatized_sent, [])
    # print("lemmatized words", lemmatized_words)

    # calculate_emotion_scores_roemolex(sum(clean_sentences, []))
    lemma_pos = get_lemmatized_words_en(translate(ro_text))
    (emotion_str, emotion_scores_str) = calculate_emotion_scores_depeche(lemma_pos)

    # print(emotion_str, emotion_scores_str)
