### 算法子图搜索引擎

> ref :  transformer [子图转换example](https://github.com/pangge/delta/blob/master/deltann/infer/example/python/simple_transformer.py)

利用Delta-infer的python API， 用户可以方便的根据tf的算法定义，把整体的算法描述的子图转换为单独的高性能算子，这里以transformer模型举例，其核心调用方式如下<sup>(详细代码参考example/python/)</sup>：

```python
import tensorflow as tf
import delta_infer as dti  # import 安装的delta infer
from standard_transformer.model import * # import 用户相关模型定义（transformer模型定义相关）

# RegistPattern 为pattern注册装饰其，name参数指定内部描述的算法被何种算子替换
# 		目前我们支持TransformerCell的高性能算子，用来替换transformer multi-head-attention计算
#     standard_transformer 函数名称并不重要，用户可以随意指定函数名称
@dti.RegistPattern(name="TransformerCell")
def standard_transformer(input_tensor = None,
                         attention_mask = None,
                         hidden_size = None,
                         num_hidden_layers = 1,
                         num_attention_heads = 12,
                         intermediate_size = 12802,
                         intermediate_act_fn = gelu,
                         hidden_dropout_prob = 0.1,
                         initializer_range = 0.02,
                         batch_size=None,
                         seq_length=None,
                         attention_head_size=None):
    with tf.variable_scope("layer_0"):
        layer_input = input_tensor
        with tf.variable_scope("attention"):
            attention_heads = []
            with tf.variable_scope("self"):
                attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=hidden_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                attention_heads.append(attention_head)
            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection. 
                attention_output = tf.concat(attention_heads, axis=-1)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            with tf.variable_scope("output"):
                attention_output = tf.layers.dense(
                                  attention_output,
                                  hidden_size,
                                  kernel_initializer=create_initializer(initializer_range))
                #attention_output = dropout(attention_output, hidden_dropout_prob)
                attention_output = layer_norm(attention_output + layer_input)

        # The activation is only applied to the "intermediate" hidden layer. 
        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
            layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
            #layer_output = dropout(layer_output, hidden_dropout_prob)
            layer_output = layer_norm(layer_output + attention_output)
    return layer_output

  
# pattern 调用
if __name__ == "__main__":
    # open graph optimizer stream, 打开一个模型优化流
    with dti.GraphStream("/path/to/your_transformer_model.pb") as gs:
        # 准备输入pattern输入信息
        batch_size = 1
        seq_length = 100
        hidden_size = 768
        num_attention_heads =12
        attention_head_size = int(hidden_size / num_attention_heads)

        # Tensor of shape [batch_size, from_seq_length, to_seq_length].
        attention_mask = tf.placeholder(tf.int32, shape=(batch_size, seq_length, seq_length))

        layer_input = tf.placeholder(tf.float32, shape=(batch_size * seq_length, hidden_size))
				# 调用注册的pattern： standard_transformer
        output_rnn = standard_transformer(input_tensor=layer_input,
                                          attention_mask=attention_mask,
                                          hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=1280,
                                          batch_size=batch_size,
                                          seq_length=seq_length,
                                          attention_head_size=attention_head_size)

        # 注册pattern的融合算子名称和关键op类型，remove in the future
        gs.register_hint_op("TransformerCell", "BatchMatMulV2")
        # 保存新的模型
        gs.save("./result.pb")
        
	  # 为了方便验证，我们可以保存注册的匹配pattern子图，这一步非必须
    with tf.compat.v1.Session() as sess:
        graph_def = dti.RegistPattern.get_patterns("TransformerCell")[0]
        with open("TransformerCell.pb", "wb") as f:
            f.write(graph_def.SerializeToString())

```



上述代码接受用户保存的训练好的model<sub>（/path/to/your_transformer_model.pb）</sub>，并根据定义的子图pattern进行搜索替换，转换Delta-infer定义的高性能算子 TransformerCell，最终生成一个新的模型 result.pb。用户可以使用这个新的model配合Delta-infer高性能算子库完成模型部署。

`Delta-infer`核心API包括：

* dti.RegistPattern(name="xxx")

  注册一个子图pattern，是一个装饰器，可以根据用户定义tf函数返回结果，确定一个pattern子图的计算方式

* dti.GraphStream("/path/to/model.pb")

  图匹配优化流，用来获取用户训练并freeze好的`model.pb`文件并进行搜索和替换

* gs.register_hint_op("custom_op_type", "hint_op_type")

  在搜索和替换的过程中，用户需要制定替换的目标函数算子类型`custom_op_type`， 并制定默认的`hint op` 。`hint_op_type`的作用是在目标图中，根据``hint_op_type`指定的类型为起始点进行搜索，`hint_op_type`需要出现在pattern中。

* Gs.save("/path/to/new_model.pb")

  设置并保存最终的优化好的图，输出的图已经经过子图替换，可以直接使用Delta-infer进行测试和部署。

  详细参考：

  > ref: [customops](https://github.com/pangge/delta/blob/master/deltann/infer/docs/customops.md)

* dti.RegistPattern.get_patterns("custom_op_type")

  获取注册的custom_op_type子图pattern，返回的是一个子图list，list的size和用户使用dti.RegistPattern注册相同的pattern 的数目一致。

  比如：

  	*  dti.RegistPattern(name="gelu")注册两种pattern，那dti.RegistPattern.get_patterns("gelu")就会返回两种子图
  	*  Delta-infer会把所有注册的子图针对目标图结构进行搜索和匹配替换，替换成相对应的`custom_op_type`





