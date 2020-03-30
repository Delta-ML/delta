import numpy as np
import tensorflow as tf
from tensorflow import graph_util
import delta_infer as dti
from nlp_transformer.model import *

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
        output, rw_head_q, mems, w_heads, tmp, AC, bd_before, BD, attn_prob, att_out, attn_vec, befornorm  = rel_multihead_attn(
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
    # add for debug output
    output = tf.layers.dense(output, 100, use_bias=False,
                               kernel_initializer=initializer, name='x')
    output = tf.layers.dense(output, 50, use_bias=False,
                               kernel_initializer=initializer, name='y')
    return output, rw_head_q, mems, w_heads, tmp, AC, bd_before, BD, attn_prob, att_out, attn_vec, befornorm

def initial_val(data):
    ret = data.flatten()
    for i in range(data.size):
        ret[i] = float(i % 10 + 1)
    return ret.reshape(data.shape)

if __name__ == "__main__":
    qlen = 100
    mlen = 16
    output = tf.compat.v1.placeholder(tf.float32, shape=(None, 4, 256 ))
    pos_emb = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, 256 )) # None = qlen+mlen
    r_w_bias = tf.compat.v1.placeholder(tf.float32, shape=(6, 64))
    r_r_bias = tf.compat.v1.placeholder(tf.float32, shape=(6, 64))
    attn_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, None)) # None = qlen+mlen
    mems = tf.compat.v1.placeholder(tf.float32, shape=(None, 4, 256))

    output_val = initial_val(np.ones((qlen, 4, 256), dtype=float))
    pos_emb_val = initial_val(np.ones((qlen+mlen, 1, 256), dtype=float))
    r_w_bias_val = initial_val(np.ones((6, 64), dtype=float))
    r_r_bias_val = initial_val(np.ones((6, 64), dtype=float))
    attn_mask_val = initial_val(np.ones((qlen, qlen+mlen), dtype=float))
    mems_val = initial_val(np.ones((mlen, 4, 256), dtype=float))

    initializer = tf.initializers.random_normal(stddev=0.02, seed=None)

    outs = TransformerCellTypeNLP(output = output,
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

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        ret, rw_head_q, mem, w_heads, tmp, AC, bd_before, BD, attn_prob, att_out, attn_vec, befornorm= sess.run(outs, {output: output_val, pos_emb: pos_emb_val, r_w_bias: r_w_bias_val, r_r_bias: r_r_bias_val, attn_mask: attn_mask_val, mems: mems_val})
        print(ret.flatten())
        print(rw_head_q.flatten())
        print(mem.flatten())
        print(w_heads.flatten())
        print(tmp.flatten())
        print(AC.flatten())
        print("bd before : ")
        print(bd_before.flatten())
        print(BD.flatten())
        print(attn_prob.flatten())
        print(np.sum(attn_prob))
        print(attn_vec.flatten())
        print(np.sum(attn_vec))
        print("norm befor: ", )
        print(befornorm.flatten())
        print(np.sum(befornorm))
        print(att_out.flatten())
        print(np.sum(att_out))
        # check first weight
        sum_col = 0.0
        for j in range(225):
            for i in range(256):
                print("get data{}: {}".format(i, tf.trainable_variables()[0].read_value().eval()[i, j]))
                sum_col += tf.trainable_variables()[0].read_value().eval()[i, j]
            break
        print(sum_col)
        #print(ff_out_before.flatten())
        # by ccw save model param
        #for node in sess.graph_def.node:
        #    print("nodename: {} ({})".format(node.name, node.op))
        graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["y/Tensordot"])#["ff/LayerNorm/batchnorm/add_1"])
        with open("nlp_fasttransformer_cell.pb", "wb") as f:
            f.write(graph_def.SerializeToString())


