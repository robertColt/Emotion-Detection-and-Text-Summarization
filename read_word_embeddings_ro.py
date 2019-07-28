import numpy as np
import pickle

word_embeddings = {}
counter = 0
total = 0
with open("data/corola.100.5.vec", encoding="utf8") as embeddings_file:
    for line in embeddings_file:
        counter += 1
        total += 1
        if len(line) < 2:
            break;
        line = line.strip().split(" ")

        word = line[0]
        coefs = np.asarray(line[1:], dtype='float32')
        word_embeddings[word] = coefs

        if counter == 100:
            print(word, coefs)
            counter = 0
        print("{}/{}".format(total, 350000))

with open("data/word_embeddings.dict", "wb") as outputfile:
    pickle.dump(word_embeddings, outputfile)

