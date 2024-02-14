from config import get_config
from data import FMRI_dataset
import pickle
import random
import numpy as np
import torch
import json
from model import Decoding_model 
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = get_config()
    print(args)
    save_name = '../results/'
    for key in args.keys():
        if key not in ['cuda']:
            save_name += key+'('+str(args[key])+')_'
    save_name = save_name[:-1]
    dataset_class = FMRI_dataset
    dataset_path = args['dataset_path']

    if 'Huth' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            print('Please specify dimension of brain input!')
        input_dataset = pickle.load(open(f'{dataset_path}{subject_name}.wq.pkl','rb'))
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Pereira' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        input_dataset = pickle.load(open(f'{dataset_path}{subject_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            print('Please specify dimension of brain input!')
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives' in args['task_name']:
        u2s = json.load(open(f'../dataset_info/u2s.json'))
        args['Narratives_stories'] = u2s[f'sub-{args["task_name"].split("_")[1]}']
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'{dataset_path}{story_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
           print('Please specify dimension of brain input!')
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)

    print('dataset initialized')
    
    if args['load_check_point']:
        decoding_model.prompt_model.check_point = decoding_model.check_point
    decoding_model.prompt_model.init_encoding_model()
    
    if args['mode'] in ['train','only_train','all','generation_train','token_train']:
        decoding_model.train(dataset.train_dataset, dataset.valid_dataset)
    # if args['mode'] in ['rank_train',]:
    #     decoding_model.train(dataset.train_dataset, dataset.train_dataset)
    if args['mode'] in ['all','evaluate','generation_train']:
        args['mode'] = 'evaluate' if args['mode'] == 'train' else args['mode']
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.test(dataset.test_dataset, args['output'])
    if args['mode'] in ['token_evaluate']:
        args['mode'] = 'evaluate' if args['mode'] == 'train' else args['mode']
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.token_test(dataset.test_dataset, args['output'])
    if args['mode'] in ['token_d']:
        pid2embedding = decoding_model.save_document_tokens(dataset.all_dataset,)
        json.dump(pid2embedding, open(decoding_model.args['checkpoint_path']+'/'+f'pid2embedding.token.json','w'))
    if args['mode'] in ['rank']:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.rank(dataset.test_dataset, args['output'])
    if args['mode'] in ['rank_train']:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.rank(dataset.train_dataset, args['output'] + '.train')
    
    if args['mode'] in ['entrophy']:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.entrophy(dataset.test_dataset, args["output"], args['output'] + '.entrophy')
