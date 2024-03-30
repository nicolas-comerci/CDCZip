import numpy as np
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
import ctypes
from typing import Union, List
import sys

os.chdir(os.path.dirname(__file__))

# MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_NAME = 'bge-base-en-v1.5'
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]

HF_PREFIX = ''
if 'all-MiniLM' in MODEL_NAME:
    HF_PREFIX = 'sentence-transformers/'
elif 'bge-' in MODEL_NAME:
    HF_PREFIX = 'BAAI/'
else:
    print(f'Unknown model name: {MODEL_NAME}')
    sys.exit(1)
N_THREADS = 32

modes = ['q4_0', 'q4_0-batchless', 'q4_1', 'f32', 'f16', 'sbert', 'sbert-batchless']

TASKS = [
    "STSBenchmark",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Get rid of the warning spam from sbert tokenizer

class EmbeddingsCppModel():
    def __init__(self, fname, fix_batch_size=-1):
        self.lib = ctypes.cdll.LoadLibrary("../build/libbert.so")


        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [ctypes.c_char_p]

        self.lib.bert_n_embd.restype = ctypes.c_int32
        self.lib.bert_n_embd.argtypes = [ctypes.c_void_p]
        
        self.lib.bert_free.argtypes = [ctypes.c_void_p]

        self.lib.bert_encode_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ]

        self.ctx = self.lib.bert_load_from_file(fname.encode("utf-8"))
        self.n_embd = self.lib.bert_n_embd(self.ctx)
        self.fix_batch_size = fix_batch_size

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        # batch_size = 8 # force batch size to 8 for now to save memory
        if self.fix_batch_size > 0:
            batch_size = self.fix_batch_size
        if isinstance(sentences, str):
            sentences = [sentences]

        n = len(sentences)

        embeddings = np.zeros((n, self.n_embd), dtype=np.float32)
        embeddings_pointers = (ctypes.POINTER(ctypes.c_float) * len(embeddings))(*[e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for e in embeddings])

        texts = (ctypes.c_char_p * n)()
        for j, sentence in enumerate(sentences):
            texts[j] = sentence.encode("utf-8")

        self.lib.bert_encode_batch(
            self.ctx, N_THREADS, batch_size, len(sentences), texts, embeddings_pointers
        )

        return embeddings

    
class BatchlessModel():
    def __init__(self, model) -> None:
        self.model = model

    def encode(self, sentences, batch_size=32, **kwargs):
        return self.model.encode(sentences, batch_size=1, **kwargs)

for mode in modes:
    if mode == 'sbert':
        model = SentenceTransformer(f"{HF_PREFIX}{MODEL_NAME}")
    elif mode == 'sbert-batchless':
        model = BatchlessModel(SentenceTransformer(f"{HF_PREFIX}{MODEL_NAME}"))
    elif mode.endswith('-batchless'):
        model = EmbeddingsCppModel(f'../models/{MODEL_NAME}/ggml-model-{mode[:-10]}.bin', fix_batch_size=1) # this work
    else:
        model = EmbeddingsCppModel(f'../models/{MODEL_NAME}/ggml-model-{mode}.bin') # this work

    evaluation = MTEB(tasks=TASKS)
    output_folder = f"results/{MODEL_NAME}_{mode}"

    evaluation.run(model, output_folder=output_folder, eval_splits=["test"], task_langs=["en"])
