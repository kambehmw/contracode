import pickle

import fire
import sentencepiece as spm
import torch
import tqdm

from models.code_moco import CodeMoCo
from models.code_mlm import CodeMLM
from representjs import RUN_DIR, CSNJS_DIR

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def embed_coco(checkpoint, data_path, spm_filepath=DEFAULT_SPM_UNIGRAM_FILEPATH):
    with open(data_path, 'rb') as f:
        matches = pickle.load(f)
    positive_samples = matches

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    n_tokens = sp.GetPieceSize()
    pad_id = sp.PieceToId("[PAD]")

    model = CodeMoCo(n_tokens=n_tokens, pad_id=pad_id)
    state = torch.load(checkpoint)
    print(state['model_state_dict'].keys())
    model.load_state_dict(state['model_state_dict'])
    model.cuda()
    model.eval()

    def make_dataset(l):
        embed_x = [torch.LongTensor(sp.EncodeAsIds(item['code'])).cuda() for item in l]
        return embed_x

    out_matches = {}
    with torch.no_grad():
        for match in tqdm.tqdm(positive_samples.keys(), desc="matches"):
            out_matches[match] = list()
            for positive in tqdm.tqdm(make_dataset(positive_samples[match])):
                try:
                    x = positive.unsqueeze(0)
                    out_matches[match].append(model.embed_x(x).cpu().numpy())
                except Exception as e:
                    print("Error!", e)
                    continue
    
    tsne_out_path = (RUN_DIR / '..' / 'tsne')
    tsne_out_path.mkdir(parents=True, exist_ok=True)
    print('writing output to ', tsne_out_path.resolve())
    with (tsne_out_path / "moco_embed.pickle").open('wb') as f:
        pickle.dump(out_matches, f)
        

def embed_bert(checkpoint, data_path, spm_filepath=DEFAULT_SPM_UNIGRAM_FILEPATH):
    with open(data_path, 'rb') as f:
        matches = pickle.load(f)
    positive_samples = matches

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    n_tokens = sp.GetPieceSize()
    pad_id = sp.PieceToId("[PAD]")

    model = CodeMLM(n_tokens=n_tokens, pad_id=pad_id)
    state = torch.load(checkpoint)
    print(state['model_state_dict'].keys())
    model.load_state_dict(state['model_state_dict'])
    model.cuda()
    model.eval()

    def make_dataset(l):
        embed_x = [torch.LongTensor(sp.EncodeAsIds(item['code'])).cuda() for item in l]
        return embed_x

    out_matches = {}
    with torch.no_grad():
        for match in tqdm.tqdm(positive_samples.keys(), desc="matches"):
            out_matches[match] = list()
            for positive in tqdm.tqdm(make_dataset(positive_samples[match])):
                try:
                    x = positive.unsqueeze(0)
                    out_matches[match].append(model.embed(x).cpu().numpy())
                except Exception as e:
                    print("Error!", e)
                    continue
    
    tsne_out_path = (RUN_DIR / '..' / 'tsne')
    tsne_out_path.mkdir(parents=True, exist_ok=True)
    print('writing output to ', tsne_out_path.resolve())
    with (tsne_out_path / "bert_embed.pickle").open('wb') as f:
        pickle.dump(out_matches, f)


if __name__ == "__main__":
    fire.Fire({"embed_coco": embed_coco, "embed_bert": embed_bert})
