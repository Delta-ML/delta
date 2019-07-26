import collections
import hashlib
import os
import subprocess
import sys
from absl import logging

dm_single_close_quote = u'\u2019' 
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote,
              ")"]

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "cnn-dailymail/url_lists/all_train.txt"
all_val_urls = "cnn-dailymail/url_lists/all_val.txt"
all_test_urls = "cnn-dailymail/url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"

num_expected_cnn_stories = 92578

VOCAB_SIZE = 200000


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files
   to a tokenized version using Stanford CoreNLP Tokenizer"""
  logging.info("Preparing to tokenize {} to {}...".format(stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  logging.info("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s),
                              os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
             '-ioFileList', '-preserveLines', 'mapping.txt']
  logging.info("Tokenizing {} files in {} and saving in {}..."
        .format(len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  logging.info("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception(
      "The tokenized stories directory {} contains {} files, but it "
      "should contain the same number as {} (which has {} files)."
      " Was there an error during tokenization?".format(
      tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  logging.info("Successfully finished tokenizing {} to {}.\n".format(stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line == "": return line
  if line[-1] in END_TOKENS: return line
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  lines = [line.lower() for line in lines]

  lines = [fix_missing_period(line) for line in lines]

  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx, line in enumerate(lines):
    if line == "":
      continue  # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  article = ' '.join(article_lines)

  abstract = ' '.join(highlights)

  return article, abstract


def write_to_file(url_file, art_out_file,
                  abs_out_file,
                  makevocab=False,
                  vocab_dir=None):
  """Reads the tokenized .story files corresponding
  to the urls listed in the url_file and writes them to a out_file."""
  logging.info("Making bin file for URLs listed in {}...".format(url_file))
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s + ".story" for s in url_hashes]
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  abs_writer = open(abs_out_file, 'w', encoding='utf8')
  art_writer = open(art_out_file, 'w', encoding='utf8')

  for idx, s in enumerate(story_fnames):
    if idx % 1000 == 0:
      logging.info("Writing story {} of {}; {:.2f} percent done"
            .format(idx, num_stories, float(idx) * 100.0 / float(num_stories)))

    # Look in the tokenized story dirs
    # to find the .story file corresponding to this url
    if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
      story_file = os.path.join(cnn_tokenized_stories_dir, s)
    else:
      continue

    article, abstract = get_art_abs(story_file)
    if len(article) == 0:
      continue

    art_writer.write(article + '\n')
    abs_writer.write(abstract + '\n')

    # Write the vocab to file, if applicable
    if makevocab:
      art_tokens = article.split(' ')
      abs_tokens = abstract.split(' ')
      abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]]
      tokens = art_tokens + abs_tokens
      tokens = [t.strip() for t in tokens]
      tokens = [t for t in tokens if t != ""]
      vocab_counter.update(tokens)

  logging.info("Finished writing file\n")

  # write vocab to file
  if makevocab:
    logging.info("Writing vocab file...")
    with open(os.path.join(vocab_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    logging.info("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception(
      "stories directory {} contains {} files but should contain {}".format(stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  if len(sys.argv) != 3:
    logging.info("USAGE: python make_datafiles.py <stories_dir> <output_dir>")
    sys.exit()
  stories_dir = sys.argv[1]
  output_dir = sys.argv[2]
  stories_dir = os.path.join(stories_dir, 'cnn/stories')
  # Check the stories directories contain the correct number of .story files
  check_num_stories(stories_dir, num_expected_cnn_stories)

  # Create some new directories
  if not os.path.exists(cnn_tokenized_stories_dir):
    os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # tokenize_stories(stories_dir, cnn_tokenized_stories_dir)
  cnn_tokenized_stories_dir = stories_dir

  set_list = ['test', 'val', 'train']
  for set_name in set_list:
    urls_set = eval('all_{}_urls'.format(set_name))
    art_path = os.path.join(output_dir, "{}.cnndm.src".format(set_name))
    abs_path = os.path.join(output_dir, "{}.cnndm.tgt".format(set_name))
    write_to_file(urls_set, art_path, abs_path)

