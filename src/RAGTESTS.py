from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

sentences = [
    "It is an animal",
    "It is a bird",
    "It is a an object"
]

sentence = [
    "Dog"
]
embeddings = model.encode(sentences)
embs = model.encode(sentence)

similarities = model.similarity(embs, embeddings)
print(similarities.shape)
print(type(similarities))
# [3, 3]