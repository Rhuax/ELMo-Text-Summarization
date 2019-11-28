import sys
import os
import json
import nltk
import collections
from tensorflow.core.example import example_pb2
import struct



dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

finished_files_dir = "finished_files_piccolo"
chunks_dir = os.path.join(finished_files_dir, "chunked")

def fix_missing_period(line):
    if line[-1] in END_TOKENS: return line
    return line + " ."



def get_art_summary(text,summary):
    # Lowercase everything
    text =text.lower()
    summary =summary.lower()
    # Make article into a single string
    text = ' '.join(nltk.word_tokenize(text))
    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    summary=f"{SENTENCE_START} {' '.join(nltk.word_tokenize(summary))} {SENTENCE_END}"
    return text, summary

def write_to_bin(dataset_jsonl_file, out_file, makevocab=False):
    if makevocab:
      vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        with open(dataset_jsonl_file,'r') as f:
            for idx,s in enumerate(f):
                obj = json.loads(s)
                if idx%1000==0:
                    print(idx)

                # Get the strings to write to .bin file
                article, abstract = get_art_summary(obj['text'],obj['summary'])

                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] # strip
                    tokens = [t for t in tokens if t!=""] # remove empty
                    vocab_counter.update(tokens)

    print(f"Finished writing file {out_file}")

  # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

def chunk_file(set_name):
    in_file = finished_files_dir+'/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk

        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]

                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'test']:#TODO VAL
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python make_datafiles.py <train> <test>")
        sys.exit()

    train = sys.argv[1]
    test = sys.argv[2]

    write_to_bin(train, os.path.join(finished_files_dir, "train.bin"),makevocab=True)
    write_to_bin(test, os.path.join(finished_files_dir, "test.bin"))
    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()