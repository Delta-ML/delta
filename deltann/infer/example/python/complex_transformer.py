import tensorflow as tf
import delta_infer as dti
from nlp_transformer.model import *

@dti.RegistPattern(name="TransformerCellNLP")
def TransformerCellTypeNLP(output=None,
                         pos_emb=None,
                         r_w_bias=None,
                         r_r_bias=None,
                         attn_mask=None,
                         mems=None,
                         d_model = 256,
                         n_head = 6,
                         d_head = 64,
                         dropout = 0.0,
                         dropatt = 0.0,
                         is_training = False,
                         initializer=None,
                         index=0,
                         d_inner=1024):
    with tf.compat.v1.variable_scope('layer_{}'.format(index)):
        output = rel_multihead_attn(
        w=output,
        r=pos_emb,
        r_w_bias=r_w_bias,
        r_r_bias=r_r_bias,
        attn_mask=attn_mask,
        mems=mems,#[index],
        d_model=d_model,
        n_head=n_head,
        d_head=d_head,
        dropout=dropout,
        dropatt=dropatt,
        is_training=is_training,
        kernel_initializer=initializer,
        index = index)
    output = positionwise_FF(
        inp=output,
        d_model=d_model,
        d_inner=d_inner,
        dropout=dropout,
        kernel_initializer=initializer,
        is_training=is_training)
    return output

if __name__ == "__main__":
    # open graph optimizer stream
    with dti.GraphStream("./model.pb") as gs:
    #with dti.GraphStream("./nlp_fasttransformer_cell.pb") as gs:
        qlen = None
        mlen = 16
        output = tf.compat.v1.placeholder(tf.float32, shape=(None, 4, 256 ))
        pos_emb = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, 256 )) # None = qlen+mlen
        r_w_bias = tf.compat.v1.placeholder(tf.float32, shape=(6, 64))
        r_r_bias = tf.compat.v1.placeholder(tf.float32, shape=(6, 64))
        attn_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, None)) # None = qlen+mlen
        mems = tf.compat.v1.placeholder(tf.float32, shape=(mlen, 4, 256))

        initializer = tf.initializers.random_normal(stddev=0.02, seed=None)

        TransformerCellTypeNLP(output = output,
                               pos_emb = pos_emb,
                               r_w_bias = r_w_bias,
                               r_r_bias = r_r_bias,
                               attn_mask = attn_mask,
                               mems = mems,
                               d_model = 256,
                               n_head = 6,
                               d_head = 64,
                               initializer = initializer,
                               d_inner = 1024)

        # remove in the future
        gs.register_hint_op("TransformerCellNLP", "BatchMatMulV2")
        gs.save("./result.pb")

    #with tf.compat.v1.Session() as sess:
    #    graph_def = dti.RegistPattern.get_patterns("TransformerCellNLP")[0]
    #    with open("TransformerCellNLP.pb", "wb") as f:
    #        f.write(graph_def.SerializeToString())
    #sess.close()
