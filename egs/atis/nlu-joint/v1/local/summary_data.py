import os
import sys
import pickle
from absl import logging

DATA_DIR = "data/"
logging.info(os.listdir(DATA_DIR))


def load_ds(fname, output_file_path):
    with open(fname, 'rb') as stream:
        ds, dicts = pickle.load(stream)
    logging.info('      samples: {:4d}'.format(len(ds['query'])))
    logging.info('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    logging.info('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    logging.info(' intent count: {:4d}'.format(len(dicts['intent_ids'])))

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
    query, slots, intent = map(ds.get,
                               ['query', 'slot_labels', 'intent_labels'])

    with open(output_file_path, "w", encoding="utf-8") as out_file:
      for i in range(len(query)):
        out_file.write(i2in[intent[i][0]]+"\t")
        out_file.write(' '.join(map(i2s.get, slots[i])) + "\t")
        out_file.write(' '.join(map(i2t.get, query[i])) + "\n")


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  if len(sys.argv) != 3:
    logging.error("Usage {} input_file output_file".format(sys.argv[0]))
    sys.exit(-1)

  output_train_file = sys.argv[1]
  output_test_file = sys.argv[2]

  load_ds(os.path.join(DATA_DIR, 'atis.train.pkl'), output_train_file)
  load_ds(os.path.join(DATA_DIR, 'atis.test.pkl'), output_test_file)
