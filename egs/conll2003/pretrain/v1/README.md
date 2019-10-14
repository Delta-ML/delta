1. You need to delete bilm/model.py 134-176 sentence after download bilm-tf project;
2. change 178 sentence as :
# concatenate the layers
lm_embeddings = tf.concat(
    [tf.expand_dims(t, axis=1) for t in layers],
    axis=1
)

return {
    'lm_embeddings': lm_embeddings,
    'lengths': lm_graph.sequence_lengths,
    'token_embeddings': lm_graph.embedding,
    'mask': lm_graph.mask,
}
