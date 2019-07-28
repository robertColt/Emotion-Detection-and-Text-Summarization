import pickle
import numpy as np

emotions = "anger anticipation disgust fear joy negative positive sadness surprise trust"
emotion_lexicon = {}
emotion_list = []


def read_roemolex():
    pass


def read_depeche_mood():
    global emotion_lexicon, emotion_list
    with open("data/DepecheMood.txt", encoding="utf8") as emotion_file:
        count = 0
        for line in emotion_file:
            if count == 0:
                count += 1
                emotion_list = line.strip().split("\t")[1:]
                continue
            # parts = word#pos range(8)... scores for each emotion
            parts = line.strip().split("\t")
            emotion_lexicon[parts[0]] = np.asarray(parts[1:], dtype="float32")
    print(emotion_lexicon["abc#n"])

read_depeche_mood()

print(len(emotion_lexicon), len(emotion_list))


with open("data/emotion_lexicon.dict", "wb") as outfile:
    pickle.dump((emotion_lexicon, emotion_list), outfile)
