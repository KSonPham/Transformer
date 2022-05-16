from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import spacy
from spacy.symbols import ORTH
import os
import torch
import warnings
warnings.filterwarnings('ignore')

class Data():
    def __init__(self,batch_size, max_length, min_freq = 2):
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_freq = min_freq
        

    def load_tokenizers(self):
        try:
            spacy_de = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            spacy_de = spacy.load("de_core_news_sm")

        try:
            spacy_en = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            spacy_en = spacy.load("en_core_web_sm")

        spacy_de.tokenizer.add_special_case(u'<s>', [{ORTH: u'<s>'}])
        spacy_de.tokenizer.add_special_case(u'</s>', [{ORTH: u'</s>'}])
        spacy_de.tokenizer.add_special_case(u'<black>', [{ORTH: u'<black>'}])
        spacy_de.tokenizer.add_special_case(u'<unk>', [{ORTH: u'<unk>'}])

        spacy_en.tokenizer.add_special_case(u'<s>', [{ORTH: u'<s>'}])
        spacy_en.tokenizer.add_special_case(u'</s>', [{ORTH: u'</s>'}])
        spacy_en.tokenizer.add_special_case(u'<black>', [{ORTH: u'<black>'}])
        spacy_en.tokenizer.add_special_case(u'<unk>', [{ORTH: u'<unk>'}])

        self.tokenizer_en = spacy_en
        self.tokenizer_de = spacy_de

    @staticmethod
    def tokenize(text, tokenizer):
            return [tok.text for tok in tokenizer.tokenizer(text)]

    def build_vocabulary(self):
        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        def tokenize_de(text):
            return Data.tokenize(text, self.tokenizer_de)

        def tokenize_en(text):
            return Data.tokenize(text, self.tokenizer_de)

        print("Getting Dataset..")
        train, val, test = Multi30k(language_pair=("de", "en"))
        self.data = [train,val,test]


        print("Building German Vocabulary..")
        self.vocab_src = build_vocab_from_iterator(
            yield_tokens(train + val + test, tokenize_de, index=0),
            min_freq=self.min_freq,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        print("Building English Vocabulary..")
        self.vocab_trg = build_vocab_from_iterator(
            yield_tokens(train + val + test, tokenize_en, index=1),
            min_freq=self.min_freq,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        self.vocab_src.set_default_index(self.vocab_src["<unk>"])
        self.vocab_trg.set_default_index(self.vocab_trg["<unk>"])
        self.padding_idx = self.vocab_src.__getitem__("<blank>")
        self.vocab_src_size = self.vocab_src.__len__()
        self.vocab_trg_size = self.vocab_trg.__len__()
        print("German Vocabulary: {} entries, English Vocabulary: {} entries".format(self.vocab_src_size,self.vocab_trg_size))

    def prepare_data_loader(self):
        def pad_to_max(tokens):
            return tokens[:self.max_length] + ["<blank>"] * max(0, self.max_length - len(tokens))

        def collate_fn(batch):
            srcs = []
            trgs = []
            for pair in batch:
                src = pair[0]
                trg = pair[1]
                # p = Data.tokenize("<s> " + src + " </s>",self.tokenizer_de)
                # c = self.vocab_src(p)

                tokenized_src = self.vocab_src(pad_to_max(Data.tokenize("<s> " + src + " </s>",self.tokenizer_de)))
                tokenized_trg = self.vocab_trg(pad_to_max(Data.tokenize("<s> " + trg + " </s>",self.tokenizer_en)))
                
                srcs.append(tokenized_src)
                trgs.append(tokenized_trg)

            srcs = torch.tensor(srcs, dtype=torch.long)
            trgs = torch.tensor(trgs, dtype=torch.long)
            return srcs, trgs

        train,val,test = self.data
        train_dataloader = DataLoader(list(train), batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(list(val), batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(list(test), batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        return train_dataloader, val_dataloader, test_dataloader

    def prepare_data(self):
        self.load_tokenizers()
        self.build_vocabulary()
        return self.prepare_data_loader()

    def get_properties(self):
        return self.padding_idx,self.vocab_src_size,self.vocab_trg_size



# if __name__ == "__main__":
#     a = Data(64,30)
    
#     train_dataloader, val_dataloader, test_dataloader = a.prepare_data()
#     b,c,d = a.get_properties()
#     for x  in train_dataloader:
#         print("1")

   
