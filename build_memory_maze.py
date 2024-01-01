import jsonlines
import sys
sys.path.append('..')
from tqdm import tqdm


fname = 'samples_zhihu.jsonl'
lang = 'Chinese'

samples = []
with jsonlines.open(fname) as reader:
    for item in reader:
        samples.append(item)
print(len(samples))

from agents.memory_tree import MemoryTree
from utils.lang_utils import split_text_into_paragraphs
from utils.file_utils import jsonl_filename_suffix
with jsonlines.open(jsonl_filename_suffix(fname, 'pro'), mode='a') as writer:

    for sample in tqdm(samples[400:]):
        if 'passage' not in sample:
            continue
        passage = sample['passage']
        sents = split_text_into_paragraphs(passage, max_length=200)
        sents = [sent.strip() for sent in sents if len(sent.strip()) > 0]
        if len(sents) <= 8:
            continue
        try:
            memory_maze = MemoryTree(lang=lang)
            memory_maze.add_sents(sents)
            sample['memory_maze'] = memory_maze.tree
        except Exception as e:
            print(e)
        writer.write(sample)
