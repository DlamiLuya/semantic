import spacy  # importing spacy

nlp = spacy.load('en_core_web_md')

# ***similarity of cat, monkey and banana***
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

#Spacy was able to deduce the similarity of the animals being cat and monkey which showed the highest similarity, 
#It was also able to deduce the connection between monkey and banana, knowing that monkeys are famous for eating the fruit.
#It also showed that cat and banana were not too similar.

#***similarity within a couple of words.***
tokens = nlp('cat apple monkey banana feline primate')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Here i added feline and primate to check if spacy would find similarity.
# A cat is of the genus feline and the monkey is of the genus primate.
# Spacy showed little similarity between the genus and the species of the animal
# What was suprising was how it was able to deduce similarity between the 2 genus feline and primate
# Which means that it knows that they belong to the same category but cannot put a species to a genus.


#***Similarity between sentences***
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#***Writting the example.py file with a simpler language model***

#I noticed firstly that the terminal showed the message 
'''The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger,
   parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models,
   e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
   You can always add your own word vectors, or use one of the larger models instead if available.
'''

# This showed me that the simpler language model couldn't take as much as the more complex language model en_core_web_md
# Upon comparing the similarity index from both i quickly saw that the simpler language model en_core_web_sm will always read,
# similarity on a lower index level compared to the more complex en_core_web_md