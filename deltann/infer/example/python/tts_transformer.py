import tensorflow as tf
import delta_infer as dti
from tts_transformer.model import *


@dti.RegistPattern(name="TransformerCell")
def TransformerCellType(src, src_mask, d_model, nhead, dim_feedforward=2048):
  output = TransformerEncoderLayer(
      d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu")(
          src, src_mask=src_mask, training=None)
  return output


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  # open graph optimizer stream
  with dti.GraphStream("./model2pb.pb") as gs:
    batch_size = 16
    seq_length = 1600
    hidden_size = 512
    nhead = 4

    #src_mask = tf.placeholder(tf.float32, shape=(batch_size, 1, seq_length, seq_length))

    src = tf.compat.v1.placeholder(
        tf.float32, shape=(batch_size, seq_length, hidden_size))

    out_trans = TransformerCellType(
        src=src,
        src_mask=None,
        d_model=hidden_size,
        nhead=nhead,
        dim_feedforward=1080)
    # remove in the future
    gs.register_hint_op("TransformerCell", "BatchMatMulV2")
    gs.save("./tts_result.pb")

  with tf.compat.v1.Session() as sess:
    graph_def = dti.RegistPattern.get_patterns("TransformerCell")[0]
    with open("TransformerCell_tts.pb", "wb") as f:
      f.write(
          tf.compat.v1.graph_util.remove_training_nodes(
              graph_def).SerializeToString())
  sess.close()
