from torch.utils.data.dataloader import default_collate
import numpy as np


def pad_collate(batch):
    max_len = float('-inf')
    # ft_model = FastText.load_fasttext_format('/home/suman/Text-VQA/data/wiki.en')
    # print(len(batch))
    lengths=[]
    for elem in batch:
        # print(elem.keys())
        embedding = np.array(elem[1])
        # print(embedding)
        # max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_len = max_len if max_len > embedding.shape[0] else embedding.shape[0]
        lengths.append(embedding.shape[0])

    lengths = np.array(lengths)
    #print(lengths)
    for i, elem in enumerate(batch):
        embedding = np.array(elem[1])
        emb_padded = np.ones(max_len, dtype=int)*38
        emb_padded[:embedding.shape[0]] = embedding

        # obj_padded = np.zeros((max_obj_num, obj_shape))
        # obj_padded[:elem['context'].shape[0]] = elem['context']
        # if question.shape[0] < max_context_len:
        #     question[]
        # _context = _context[-max_context_len:]
        # context = np.zeros((max_context_len, max_context_sen_len))
        # for j, sen in enumerate(_context):
        #     context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        # question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        # q_vec = []
        # text_vec = []
        # for q in question:
        #     q_vec.append(get_ft_vec(q, ft_model))
        # for text in elem['scene_texts']:
        #     text_vec.append(get_ft_vec(text, ft_model))

        batch[i] = (elem[0], emb_padded, lengths[i])

    return default_collate(batch)
