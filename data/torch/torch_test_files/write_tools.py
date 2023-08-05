import pymongo

"""
You should configure the database
"""
torch_db = pymongo.MongoClient(host="localhost", port=27017)["Torch-VulFuzz"]

def write_fn(func_name, params, input_signature, output_signature):
    params = dict(params)
    out_fname = "torch." + func_name
    params['input_signature'] = input_signature
    params['output_signature'] = output_signature
    params['source'] = 'tests'
    torch_db[out_fname].insert_one(params)
   
	
