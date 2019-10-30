class Sentence(object):
    def __init__ (self, words, seq_chars, tags, word_ids, seq_char_ids, tag_ids, edu_ids):
        self.words = words
        self.seq_chars = seq_chars
        self.tags = tags
        self.word_ids = word_ids
        self.seq_char_ids = seq_char_ids
        self.tag_ids = tag_ids
        self.edu_ids = edu_ids

    def length(self):
        return len(self.words)

class Instance(object):
    def __init__(self, sentences, syntax_features):
        self.edus = []
        
        cur_edu_id = 1
        cur_words = []
        cur_tags = []
        cur_syntax = []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            syntax = syntax_features[idx]
            for idy in range(len(sentence.words)):
                if sentence.edu_ids[idy] != cur_edu_id:
                    cur_edu_id += 1
                    self.edus.append(EDU(cur_words, cur_tags, '<S>', cur_syntax))
                    cur_words = []
                    cur_tags = []
                    cur_syntax = []
                cur_words.append(sentence.words[idy])
                cur_tags.append(sentence.tags[idy])
                cur_syntax.append(syntax[:,idy,:])
        self.edus.append(EDU(cur_words, cur_tags, '<P>', cur_syntax))

class EDU(object):
    def __init__(self, words, tags, etype, syntax_features):
        self.words = words
        self.tags = tags
        self.etype = etype
        self.syntax_features = syntax_features
            
