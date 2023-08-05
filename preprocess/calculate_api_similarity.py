from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pymongo, json
import torch
from torch.nn import CosineSimilarity
from scipy.spatial.distance import cosine
DB = pymongo.MongoClient(host="localhost", port=27017)["orion-tf1"]

def run():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    all_apis = DB.list_collection_names()
    while all_apis:
        current_api = all_apis.pop()
        list_of_sim = []
        for j, api_ in enumerate(all_apis):
            print(f"{api_}: Index: {j}")
            embedding1 = model.encode(current_api)
            embedding2 = model.encode(api_)
            cosine_sim = cosine_similarity([embedding1], [embedding2])
            list_of_sim.append([api_, float(cosine_sim[0][0])])
        sorted_list = sorted(list_of_sim, key=lambda x: x[1], reverse=True)
        for item in sorted_list:
            json_data = json.dumps(item)
            with open(f'/media/nimashiri/DATA/vsprojects/benchmarkingDLFuzzers/fuzzers/DeepREL/tensorflow/data/tf/match_internal/{current_api}.json', 'a') as f:
                f.write(json_data)
                f.write('\n')

if __name__ == '__main__':
    run()