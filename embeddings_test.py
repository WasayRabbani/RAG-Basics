from google import genai
from dotenv import load_dotenv

load_dotenv()

client=genai.Client()



"""client         →  Gemini se connection
.models           →  inn models mein se koi chahiye
.embed_content    →  embedding karna hai content ka
(model=...)       →  konsa model use karo
(contents=...)    →  kaunsa text embed karo"""


# result=client.models.embed_content(
#     model="gemini-embedding-001",
#     contents='This is my first embedding'   
# )

"""
So here we got the embeddings for this sentence.
One thing.
1. Same model, same embedding, gemini-001 and gemini-002 gives different embeddings as both have different embeddings.
OpenAI different. 
In whole project use same model.
"""

e1=client.models.embed_content(
    
    model='gemini-embedding-001',
    contents=['I love football', "Soccer is my passion" ,"I enjoy cooking"]
)

print(type(e1.embeddings[0].values))

# Checking Cosine Similarity.

from sklearn.metrics.pairwise import cosine_similarity
cs=cosine_similarity()


print(cs(X=[e1.embeddings[1].values],Y=[e1.embeddings[2].values]))
