import jsonlines
import sys
sys.path.append('..')
from tqdm import tqdm
samples = []
with jsonlines.open('samples_kiss.jsonl') as reader:
    for item in reader:
        samples.append(item)
print(len(samples))

from agents.memory_tree import MemoryTree
with jsonlines.open('samples_kiss_pro.jsonl', mode='a') as writer:
    for sample in tqdm(samples[400:]):
        if 'passage' not in sample:
            continue
        passage = sample['passage']
        sents = passage.split('\n')
        sents = [sent.strip() for sent in sents if len(sent.strip()) > 0]
        if len(sents) <= 80:
            continue
        try:
            memory_maze = MemoryTree(lang='English')
            memory_maze.add_sents(sents)
            sample['memory_maze'] = memory_maze.tree
        except Exception as e:
            print(e)
        writer.write(sample)
