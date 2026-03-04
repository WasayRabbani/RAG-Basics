import faiss



dimension=3072

# Index means actually the whole place that coontains all the embeddings.
# Flat means compare new embedding with all available embeddings
# L2: a similarity test almost same as cosine similarity 


# Part 1: Making Index (container)
container=faiss.IndexFlatL2(dimension)
print(container.ntotal)

# Part 2: Add embeddings to container (Index)
from google import genai
from dotenv import load_dotenv

load_dotenv()

client=genai.Client()


emb=client.models.embed_content(
    model='gemini-embedding-001',
    contents=['My name is Wasay','Cat is inside the house','Dog is outside','Wasay is in 8th semester']
)

# print(emb.embeddings[0].values[:5])
    
# Part 3: Storing Embeddings in FAISS container

import numpy as np

result=[]
for i in emb.embeddings:
    result.append(i.values)

vector=np.array(
    result,dtype=np.float32
)

# Adding this vector to FAISS container (index)
container.add(vector)



# Asking question and searching in FAISS
query='In which semester Wasay is?'

emb_question=client.models.embed_content(
    model='gemini-embedding-001',
    contents=query
)


   
query_vector=np.array([
    emb_question.embeddings[0].values],
    dtype=np.float32
)


distance,index=container.search(query_vector,k=2)
print(f'{distance}\n{index}')




