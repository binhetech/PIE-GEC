# code adapted from https://github.com/google-research/bert

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import collections
import pickle
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow.python.lib.io.file_io import get_matching_files
from tensorflow.contrib.distribute import AllReduceCrossDeviceOps

import modeling
# obtains contextual embeddings for appends and replacements for edit factorized architecture figure 2 in the paper
import modified_modeling
import optimization
import tokenization
import custom_optimization
import wem_utils

flags = tf.flags

## Required parameters
flags.DEFINE_string("data_dir", None,
                    "The input data dir. Should contain the .txt files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
# 模型检查点保存的输出目录
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")
## Other parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", False,
                  "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 128,
                     "The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
# 训练参数
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 512, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 512, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
# 在warm up热身阶段执行线性学习率进行训练的样本比例:10%
flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
# 保存模型检查点的频率：1000steps
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
# 设置是否使用TPU，或者GPU训练
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("use_gpu", True, "Whether to use GPU.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_integer("n_gpus", 2, "Only used if `use_gpu` is True. Total number of GPU cores to use.")
flags.DEFINE_string("tpu_name", None,
                    "The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_zone", None,
                    "[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("gcp_project", None,
                    "[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_float("copy_weight", 1, "weight to copy")
flags.DEFINE_bool("use_bert_more", True, "use bert more exhaustively for logit computation")
flags.DEFINE_string("path_inserts", None, "path to insert pickle")
flags.DEFINE_string("path_multitoken_inserts", None, "path to multitoken_inserts")
flags.DEFINE_bool("subtract_replaced_from_replacement", True, "subtract_replaced_from_replacement")
flags.DEFINE_string("eval_checkpoint", None, "checkpoint to evaluate gec model")
flags.DEFINE_string("predict_checkpoint", None, "checkpoint to use for predictions")
flags.DEFINE_integer("random_seed", 0, "random seed for creating random initializations")
flags.DEFINE_bool("create_train_tf_records", True, "whether to create train tf records")
flags.DEFINE_bool("create_predict_tf_records", True, "whether to create predict tf records")
# flags.DEFINE_bool("dump_probs", False, "dump edit probs to numpy file while decoding")

FLAGS = flags.FLAGS


class PaddingInputExample(object):
    """
    Fake example so the num input examples is a multiple of the batch size.
    一个假的填充输入样本类。用于填充虚假样本，主要因为在TPU上进行eval/predict时，样本数必须是batch size的整数

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class GECInputExample(object):
    """
    GEC输入样本类，用于构造输入样本对象.

    本类中包含以下方法:
        __init__: 初始化方法

    """

    def __init__(self, guid, input_sequence, edit_sequence=None):
        """
        初始化方法.

        Args:
            guid: string, 样本id标识, 如: train-0, dev-1, test-2, ...
            input_sequence: list of list(int), 输入特征序列X
            edit_sequence: list of list(int), 对应输入的编辑序列，即标签序列Y

        Return:
            None

        """
        # guid = "%s-%s" % (set_type, i)
        self.guid = guid
        self.input_sequence = input_sequence
        # None when test set
        self.edit_sequence = edit_sequence


class GECInputFeatures(object):
    """
    GEC输入特征类.

    本类中包含以下方法:
        __init__: 初始化方法

    """

    def __init__(self, input_sequence, input_mask, segment_ids, edit_sequence):
        """

        初始化方法.

        Args:
            input_sequence: list of int, 输入特征序列X
            input_mask: list of int, 输入遮盖位置id
            segment_ids: list of int, 输入的分段id
            edit_sequence: list of int, 标签序列y

        Return:
            None

        """
        self.input_sequence = input_sequence
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.edit_sequence = edit_sequence


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            return (line for line in f)


class GECProcessor(DataProcessor):
    """
    GEC数据处理类，用于读取、加载样本数据、创建输入样本对象.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        train_incorr = self._read_file(os.path.join(data_dir, "train_incorr.txt"))
        train_labels = self._read_file(os.path.join(data_dir, "train_labels.txt"))
        return self._create_examples(train_incorr, train_labels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_incorr = self._read_file(os.path.join(data_dir, "dev_incorr.txt"))
        dev_labels = self._read_file(os.path.join(data_dir, "dev_labels.txt"))
        return self._create_examples(dev_incorr, dev_labels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_incorr = self._read_file(os.path.join(data_dir, "test_incorr.txt"))
        # test_labels = self._read_file(os.path.join(data_dir, "test_labels.txt"))
        test_labels = None
        return self._create_examples(test_incorr, test_labels, "test")

    def _create_examples(self, incorr_lines, labels_lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type != "test":
            # train / dev set
            for (i, (incorr_line, labels_line)) in enumerate(zip(incorr_lines, labels_lines)):
                guid = "%s-%s" % (set_type, i)
                input_sequence = incorr_line
                edit_sequence = labels_line
                # 使用yield可以避免一次性加载大文件，解决内存超出问题
                yield GECInputExample(guid, input_sequence, edit_sequence)
        else:
            # test set, 测试数据集，没有标签数据labels_lines
            for (i, incorr_line) in enumerate(incorr_lines):
                guid = "%s-%s" % (set_type, i)
                input_sequence = incorr_line
                edit_sequence = None
                yield GECInputExample(guid, input_sequence, edit_sequence)


def sequence_padding(input_sequence, edit_sequence, max_seq_length):
    input_sequence = list(map(int, input_sequence))
    if len(input_sequence) > max_seq_length:
        # 输入序列长度超出设置的最大序列长度后，直接截断
        input_sequence = input_sequence[0:(max_seq_length)]

    if edit_sequence:
        # train or dev set
        edit_sequence = list(map(int, edit_sequence.strip().split()))
        if len(edit_sequence) > max_seq_length:
            edit_sequence = edit_sequence[0:(max_seq_length)]

        if len(input_sequence) != len(edit_sequence):
            print("This should ideally not happen")
            exit(1)
    else:
        # test set 没有标签
        edit_sequence = None

    # 有数据的输入遮盖初始化为1，而后面填充的值为0
    input_mask = [1] * len(input_sequence)
    # 分段id全部为0
    segment_ids = [0] * len(input_sequence)

    # Zero-pad up to the sequence length. 当输入序列<最大序列长度时，需要填充0值
    while len(input_sequence) < max_seq_length:
        input_sequence.append(0)
        if edit_sequence:
            edit_sequence.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    if not edit_sequence:
        # 对于test set而言，标签序列全部初始化为-1
        edit_sequence = [-1] * max_seq_length

    assert len(input_sequence) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(edit_sequence) == max_seq_length

    return input_sequence, input_mask, segment_ids, edit_sequence


def example_padding(example, max_seq_length):
    return sequence_padding(example.input_sequence.strip().split(), example.edit_sequence, max_seq_length)


def gec_convert_single_example(example, max_seq_length, ex_index):
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    将一个输入样本对象转换成为一个输入特征对象.

    Args:
        ex_index: int, 前一个样本对象的索引
        example: class, 输入样本对象
        max_seq_length: int, 最大序列sequence的长度

    """
    if isinstance(example, PaddingInputExample):
        # 如果输入样本类为虚假的填充输入样本对象，则直接返回全部0值的输入特征对象
        return GECInputFeatures(
            input_sequence=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            edit_sequence=[0] * max_seq_length)

    input_sequence, input_mask, segment_ids, edit_sequence = example_padding(example, max_seq_length)

    if ex_index < 5:
        # 对于前5个样本，打印日志
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("input_sequence: %s" % " ".join([str(x) for x in input_sequence]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("edit_sequence: %s" % " ".join([str(x) for x in edit_sequence]))

    # 构建输入特征对象
    feature = GECInputFeatures(
        input_sequence=input_sequence,
        input_mask=input_mask,
        segment_ids=segment_ids,
        edit_sequence=edit_sequence)
    return feature


def gec_file_based_convert_examples_to_features(examples, max_seq_length, output_dir, mode, num_examples):
    """
    Convert a set of `InputExample`s to a TFRecord file.
    基于输入文件将样本对象转为为特征对象，并写入TFRecord格式文件.

    Args:
        examples: list of class, 样本对象列表
        max_seq_length: int, 最大序列长度
        output_dir: string, 输出文件夹
        mode: string, 数据集类型，包括："train", "eval", "predict"
        num_examples: int, 样本个数

    Return:
        None, 写入TFRecord file: {mode}_{num}.tf_record

    """
    num_writers = 0
    writer = None
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            # 每10000个样本时打印当前日志
            tf.logging.info("Writing example %d of %d" % (ex_index, num_examples))
        if ex_index % 500000000000 == 0:
            if writer:
                writer.close()
                del writer
            output_file = os.path.join(output_dir, "{}_{}.tf_record".format(mode, num_writers))
            # 初始化一个文件写入对象，每5000亿个样本就重新构建一次，分别存储成多个文件
            writer = tf.python_io.TFRecordWriter(output_file)
            num_writers += 1

        # 将一个样本对象转换为一个特征对象
        feature = gec_convert_single_example(example, max_seq_length, ex_index)

        def create_int_feature(values):
            # 数据格式转换为int64
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # 特征数据：即有序词典
        features = collections.OrderedDict()
        features["input_sequence"] = create_int_feature(feature.input_sequence)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["edit_sequence"] = create_int_feature(feature.edit_sequence)

        # 将特征数据转换成为tf.train.Example
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # 序列化写入文件
        writer.write(tf_example.SerializeToString())


def gec_file_based_input_fn_builder(output_dir, mode, max_seq_length,
                                    is_training, drop_remainder):
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.
    基于文件的输入函数input function构建类.

    Args:
        output_dir: string, 输出文件夹
        mode: string, 数据集类型，包括："train", "eval", "predict"
        max_seq_length: int, 最大序列长度
        is_training: boolean, 是否是训练模式
        drop_remainder: boolean, 是否去掉最后多余的样本example。训练时，去掉；而评估或者预测时，只有在TPU上运算时才去掉；

    """
    # 根据tf_record文件后缀匹配输入数据
    input_files = get_matching_files(output_dir + "/" + "{}_*.tf_record".format(mode))
    print("INPUT_FILES: " + " AND ".join(input_files))

    name_to_features = {
        "input_sequence": tf.FixedLenFeature([max_seq_length], tf.int64),  # 固定长度的tensor
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "edit_sequence": tf.FixedLenFeature([max_seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """
        Decodes a record to a TensorFlow example.
        将一个tf record转换成为一个tf.Example.
        """
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        if FLAGS.use_tpu and FLAGS.tpu_name:
            # 把所有int64数据转为int32, 因为TPU只支持tf.int32
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
        return example

    def input_fn(params):
        """
        The actual input function. 输入函数
        """
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_files)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=5000)
        d = d.apply(tf.contrib.data.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                  batch_size=batch_size, drop_remainder=drop_remainder))
        return d

    return input_fn


def edit_word_embedding_lookup(embedding_table, input_ids, use_one_hot_embeddings, vocab_size, embedding_size):
    """
    词嵌入查找.

    :param embedding_table:
    :param input_ids:
    :param use_one_hot_embeddings: boolean, 是否使用one-hot embedding
    :param vocab_size:
    :param embedding_size:
    :return:
    """
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return output


def gec_create_model(bert_config, is_training, input_sequence,
                     input_mask, segment_ids, edit_sequence,
                     use_one_hot_embeddings, mode,
                     copy_weight,
                     use_bert_more,
                     insert_ids,
                     multitoken_insert_ids,
                     subtract_replaced_from_replacement):
    """
    Creates a classification model.
    构建GEC分类模型类.

    Args:
        bert_config: bert配置json文件
        is_training: boolean, 是否训练模式
        input_sequence: list of int
        input_mask: list of int
        segment_ids
        edit_sequence:
        use_one_hot_embeddings:
        mode:
        copy_weight:
        use_bert_more
        insert_ids: word ids of unigram inserts (list)
        multitoken_insert_ids: word_ids of bigram inserts (list of tuples of length 2)
        subtract_replaced_from_replacement:

    """
    # insert_ids:
    # multitoken_insert_ids:
    # Defining the space of all possible edits:
    # unk, sos and eos are dummy edits mapped to 0, 1 and 2 respectively
    # copy is mapped to 3
    # del is mapped to 4
    num_appends = len(insert_ids) + len(multitoken_insert_ids)
    num_replaces = num_appends  # appends and replacements come from the same set (inserts and multitoken_inserts)
    append_begin = 5  # First append edit (mapped to 5)
    append_end = append_begin + num_appends - 1  # Last append edit
    rep_begin = append_end + 1  # First replace edit
    rep_end = rep_begin + num_replaces - 1  # Last replace edit
    num_suffix_transforms = 58  # num of transformation edits
    num_labels = 5 + num_appends + num_replaces + num_suffix_transforms  # total number of edits
    print("************ num of labels : {} ***************".format(num_labels))

    config = bert_config
    input_sequence_shape = modeling.get_shape_list(input_sequence, 2)
    batch_size = input_sequence_shape[0]
    seq_len = input_sequence_shape[1]

    if not use_bert_more:  # default use of bert (without logit factorisation)
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_sequence,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        output_layer = model.get_sequence_output()
    else:  # LOGIT FACTORISATION is On!
        model = modified_modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_sequence,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_sequence_output()
        # representation of replacement slots as described in paper
        replace_layer = output_layer[:, seq_len:2 * seq_len, :]
        # representation of append slots as described in paper
        append_layer = output_layer[:, 2 * seq_len:3 * seq_len, :]
        output_layer = output_layer[:, 0:seq_len, :]

    output_layer_shape = modeling.get_shape_list(output_layer, 3)
    hidden_size = output_layer_shape[-1]

    flattened_output_layer = tf.reshape(output_layer, [-1, hidden_size])

    h_edit = flattened_output_layer

    if use_bert_more:
        h_word = flattened_output_layer
        flattened_replace_layer = tf.reshape(replace_layer, [-1, hidden_size])
        flattened_append_layer = tf.reshape(append_layer, [-1, hidden_size])

        m_replace = flattened_replace_layer
        m_append = flattened_append_layer

        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
                h_word = tf.layers.dense(
                    h_word,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                h_word = modeling.layer_norm(h_word)

        with tf.variable_scope("cls/predictions", reuse=True):
            with tf.variable_scope("transform", reuse=True):
                m_replace = tf.layers.dense(
                    m_replace,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                m_replace = modeling.layer_norm(m_replace)

        with tf.variable_scope("cls/predictions", reuse=True):
            with tf.variable_scope("transform", reuse=True):
                m_append = tf.layers.dense(
                    m_append,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                m_append = modeling.layer_norm(m_append)

        word_embedded_input = model.word_embedded_input
        flattened_word_embedded_input = tf.reshape(word_embedded_input, [-1, hidden_size])

    labels = edit_sequence
    edit_weights = tf.get_variable("edit_weights", [num_labels, hidden_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))

    if is_training:
        h_edit = tf.nn.dropout(h_edit, keep_prob=0.9)

    if use_bert_more:
        # append/replace weight vector for a given append or replace operation
        # correspond to word embedding for its token argument
        # for multitoken append/replace (e.g. has been)
        # weight vector is sum of word embeddings of token arguments

        append_weights = edit_word_embedding_lookup(model.embedding_table, insert_ids,
                                                    use_one_hot_embeddings, config.vocab_size, config.hidden_size)
        replace_weights = append_weights  # tokens in replace and append vocab are same
        # (i.e. inserts and multitoken_inserts)

        multitoken_append_weights = wem_utils.edit_embedding_lookup(model.embedding_table, multitoken_insert_ids,
                                                                    use_one_hot_embeddings, config.vocab_size,
                                                                    config.hidden_size)
        multitoken_replace_weights = multitoken_append_weights  # tokens in replace and append vocab are same
        # (i.e. inserts and multitoken_inserts)

        append_weights = tf.concat([append_weights, multitoken_append_weights], 0)
        replace_weights = tf.concat([replace_weights, multitoken_replace_weights], 0)

    # 计算损失loss
    with tf.variable_scope("loss"):
        edit_logits = tf.matmul(h_edit, edit_weights, transpose_b=True)  # first term in eq3 in paper
        logits = edit_logits
        if use_bert_more:
            # =============== inplace_word_logits==============# #2nd term in eq3 in paper
            inplace_logit = tf.reduce_sum(h_word * flattened_word_embedded_input, axis=1, keepdims=True)  # copy
            # inplace_logit = tf.reduce_sum(m_replace * flattened_word_embedded_input, axis=1, keepdims=True) #copy
            inplace_logit_appends = tf.tile(inplace_logit, [1, num_appends])
            inplace_logit_transforms = tf.tile(inplace_logit, [1, num_suffix_transforms])
            zero_3_logits = tf.zeros([batch_size * seq_len, 3])  # unk sos eos
            zero_1_logits = tf.zeros([batch_size * seq_len, 1])  # del
            zero_replace_logits = tf.zeros([batch_size * seq_len, num_replaces])

            concat_list = [zero_3_logits, inplace_logit, zero_1_logits] \
                          + [inplace_logit_appends] \
                          + [zero_replace_logits] \
                          + [inplace_logit_transforms]

            inplace_word_logits = tf.concat(concat_list, 1)

            # ======additional (insert,replace) logits ====# #3rd term in eqn3 in paper
            zero_5_logits = tf.zeros([batch_size * seq_len, 5])
            append_logits = tf.matmul(m_append, append_weights, transpose_b=True)

            if subtract_replaced_from_replacement:
                replace_logits = replacement_minus_replaced_logits(m_replace,
                                                                   flattened_word_embedded_input, replace_weights)
            else:
                replace_logits = tf.matmul(m_replace, replace_weights, transpose_b=True)

            suffix_logits = tf.zeros([batch_size * seq_len, num_suffix_transforms])

            concat_list = [zero_5_logits, append_logits, replace_logits, suffix_logits]
            additional_logits = tf.concat(concat_list, 1)
            # ====================================================#

            logits = edit_logits + inplace_word_logits + additional_logits
            logits_bias = tf.get_variable("output_bias", shape=[num_labels], initializer=tf.zeros_initializer())
            logits += logits_bias

        logits = tf.reshape(logits, [output_layer_shape[0], output_layer_shape[1], num_labels])
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.nn.softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_token_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        per_token_loss = per_token_loss * tf.to_float(input_mask)
        mask = copy_weight * tf.to_float(tf.equal(labels, 3)) + tf.to_float(tf.not_equal(labels, 3))
        masked_per_token_loss = per_token_loss * mask
        per_example_loss = tf.reduce_sum(masked_per_token_loss, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probs)


def replacement_minus_replaced_logits(replace_layer, word_embedded_input, weights):
    result_1 = tf.matmul(replace_layer, weights, transpose_b=True)
    result_2 = replace_layer * word_embedded_input
    result_2 = tf.reduce_sum(result_2, 1)
    result_2 = tf.expand_dims(result_2, -1)
    return result_1 - result_2


def gec_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu,
                         use_one_hot_embeddings, copy_weight,
                         use_bert_more,
                         inserts, insert_ids,
                         multitoken_inserts, multitoken_insert_ids,
                         subtract_replaced_from_replacement):
    """
    Returns `model_fn` closure for TPUEstimator.
    模型函数类. 用于创建估计器estimator

    Args:
        bert_config
        init_checkpoint
        learning_rate
        num_train_steps
        num_warmup_steps
        use_tpu
        use_one_hot_embeddings
        copy_weight
        use_bert_more
        inserts
        insert_ids
        multitoken_inserts
        multitoken_insert_ids
        subtract_replaced_from_replacement

    Return:
        tf.estimator.EstimatorSpec实例

    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        """
        features： This is batch_features from input_fn
        labels： This is batch_labels from input_fn
        mode：   An instance of tf.estimator.ModeKeys
        params： Additional configuration
        """

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_sequence = features["input_sequence"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        edit_sequence = features["edit_sequence"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = gec_create_model(
            bert_config, is_training, input_sequence,
            input_mask, segment_ids, edit_sequence,
            use_one_hot_embeddings, mode,
            copy_weight,
            use_bert_more,
            insert_ids,
            multitoken_insert_ids,
            subtract_replaced_from_replacement)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            # 如果初始化检查点文件
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            if FLAGS.use_tpu and FLAGS.tpu_name:
                # TPU train
                train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                         use_tpu)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                # GPUs or CPU train
                train_op = custom_optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, edit_sequence, logits):
                predictions = tf.argmax(logits[:, :, 3:], axis=-1, output_type=tf.int32) + 3
                mask = tf.equal(edit_sequence, 0)
                mask = tf.logical_or(mask, tf.equal(edit_sequence, 1))
                mask = tf.logical_or(mask, tf.equal(edit_sequence, 2))
                mask = tf.logical_or(mask, tf.equal(edit_sequence, 3))
                mask = tf.to_float(tf.logical_not(mask))
                accuracy = tf.metrics.accuracy(edit_sequence, predictions, mask)
                loss = tf.metrics.mean(per_example_loss)
                result_dict = {}
                result_dict["eval_accuracy"] = accuracy
                result_dict["eval_loss"] = loss
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, edit_sequence, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            # first three edit ids unk, sos, eos are dummy. We do not consider them in predictions
            predictions = tf.argmax(logits[:, :, 3:], axis=-1, output_type=tf.int32) + 3
            if FLAGS.use_tpu and FLAGS.tpu_name:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"predictions": predictions, "logits": logits},
                    scaffold_fn=scaffold_fn)
            else:
                # multiple GPUs
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"predictions": predictions, "logits": logits})
        return output_spec

    return model_fn


def get_file_length(file_address):
    """
    计算文件长度.

    Args:
        file_address: string, 文件路径

    Return: int, 文件长度

    """
    num_lines = sum(1 for _ in tf.gfile.GFile(file_address, "r"))
    return num_lines


def main(_):
    # 设置日志级别为INFO
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    # 创建输出目录
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # bert配置文件
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" % (
                FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 构建GEC数据处理器
    processor = GECProcessor()

    # 构建tokenizer对象
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # 读入文件：插入编辑tokens
    inserts = pickle.load(tf.gfile.Open(FLAGS.path_inserts, "rb"))
    insert_ids = tokenizer.convert_tokens_to_ids(inserts)
    # 读取：多token插入编辑
    multitoken_inserts = pickle.load(tf.gfile.Open(FLAGS.path_multitoken_inserts, "rb"))
    multitoken_insert_ids = wem_utils.list_to_ids(multitoken_inserts, tokenizer)

    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        # 设置随机种子数
        tf.set_random_seed(FLAGS.random_seed)
        # 总训练样本数
        num_train_examples = get_file_length(os.path.join(FLAGS.data_dir, "train_labels.txt"))
        print("Number of training examples: {}".format(num_train_examples))

        # 计算训练总步数=[总训练样本数 mod 训练batch size]*训练的epoch轮数
        num_train_steps = int((num_train_examples / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
        # 计算warm up步数。 warmup和decay的Adam（AdamWeightDecayOptimizer），这两个参数策略用于动态学习率
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 构造模型输入函数：model_fn，用于构造估计器estimator
    model_fn = gec_model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        copy_weight=FLAGS.copy_weight,
        use_bert_more=FLAGS.use_bert_more,
        inserts=inserts,
        insert_ids=insert_ids,
        multitoken_inserts=multitoken_inserts,
        multitoken_insert_ids=multitoken_insert_ids,
        subtract_replaced_from_replacement=FLAGS.subtract_replaced_from_replacement, )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu and FLAGS.tpu_name:
        # 1.创建tpu集群
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone,
                                                                              project=FLAGS.gcp_project)
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        # 2.设置run_config
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=15,
            tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                                num_shards=FLAGS.num_tpu_cores,
                                                per_host_input_for_training=is_per_host),
        )
        # 3.构造TPU评估器对象，用于TPU训练
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,  # 是否使用tpu, boolean
            model_fn=model_fn,  # 模型函数
            config=run_config,  # 运行配置
            train_batch_size=FLAGS.train_batch_size,  # 64
            eval_batch_size=FLAGS.eval_batch_size,  # 521
            predict_batch_size=FLAGS.predict_batch_size)  # 512
    else:
        # 1.先定义分布式训练的镜像策略：MirroredStrategy
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.n_gpus,  # 使用gpu的个数
            cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=FLAGS.n_gpus),  # 各设备之间的数据操作方式
            # cross_device_ops=AllReduceCrossDeviceOps('hierarchical_copy'),
        )
        # 2.设置会话session配置
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        # 3.设置运行配置run_config
        run_config = tf.contrib.tpu.RunConfig(
            train_distribute=dist_strategy,
            eval_distribute=dist_strategy,
            model_dir=FLAGS.output_dir,
            session_config=session_config,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=15)
        # 4.构造CPU、GPU评估器对象
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": FLAGS.train_batch_size})

    # train on train set在训练集上进行训练
    if FLAGS.do_train:
        train_record_dir = FLAGS.output_dir
        if FLAGS.create_train_tf_records:
            # 1、基于文件将样本对象转换为特征对象，并存储TF Records格式的文件
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            gec_file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, train_record_dir, "train",
                                                        num_train_examples)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # 2、构建train的特征输入函数input function
        train_input_fn = gec_file_based_input_fn_builder(output_dir=train_record_dir, mode="train",
                                                         max_seq_length=FLAGS.max_seq_length, is_training=True,
                                                         drop_remainder=True)
        # 3、评估器train训练
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # evaluate on dev set在开发集上进行评估
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_eval_examples = get_file_length(os.path.join(FLAGS.data_dir, "dev_labels.txt"))
        # 1、基于文件将样本对象转换为特征对象，并存储TF Records格式的文件
        gec_file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length, FLAGS.output_dir, "eval",
                                                    num_eval_examples)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # However, if running eval on the TPU, you will need to specify the
        # number of steps. 只有使用TPU时才需要steps, 其他的GPUs或者CPU运算不需要，直到输入结束、
        if FLAGS.use_tpu:
            eval_steps = int(num_eval_examples / FLAGS.eval_batch_size)
            # Eval will be slightly WRONG on the TPU because it will truncate the last batch.
            eval_drop_remainder = True
        else:
            eval_steps = None
            eval_drop_remainder = False
        # 2、构建eval的特征输入函数input function
        eval_input_fn = gec_file_based_input_fn_builder(output_dir=FLAGS.output_dir, mode="eval",
                                                        max_seq_length=FLAGS.max_seq_length, is_training=False,
                                                        drop_remainder=eval_drop_remainder)
        # 3、评估器evaluate评估，返回结果
        result = estimator.evaluate(input_fn=eval_input_fn, checkpoint_path=FLAGS.eval_checkpoint, steps=eval_steps)
        # 4、输出评估集的结果
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    # predict on test set在测试集上进行预测
    if FLAGS.do_predict:
        # 只计算测试样本个数: data_dir/test_incorr.txt
        num_test_examples = get_file_length(os.path.join(FLAGS.data_dir, "test_incorr.txt"))
        print("num of test_examples: {}".format(num_test_examples))
        num_actual_predict_examples = num_test_examples

        if FLAGS.create_predict_tf_records:
            # 创建tf records
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
            if FLAGS.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                # raise ValueError("Prediction in TPU not supported")
                padded_examples = []
                # 计算需要padding填充的样本个数
                while num_test_examples % FLAGS.predict_batch_size != 0:
                    padded_examples.append(PaddingInputExample())
                    num_test_examples += 1
                # 填充虚假fake样本
                iterables = [predict_examples, padded_examples]
                predict_examples = chain()
                for iterable in iterables:
                    predict_examples = chain(predict_examples, iterable)

            # 1、基于文件将样本对象转换为特征对象，并存储TF Records格式的文件
            gec_file_based_convert_examples_to_features(predict_examples, FLAGS.max_seq_length, FLAGS.output_dir,
                                                        "predict", num_test_examples)

            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)", num_test_examples,
                            num_actual_predict_examples, num_test_examples - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        # 2、构建特征输入函数
        predict_input_fn = gec_file_based_input_fn_builder(output_dir=FLAGS.output_dir, mode="predict",
                                                           max_seq_length=FLAGS.max_seq_length, is_training=False,
                                                           drop_remainder=predict_drop_remainder)
        # 3、评估器进行预测
        result = estimator.predict(input_fn=predict_input_fn, checkpoint_path=FLAGS.predict_checkpoint)
        print("type of result: {}".format(type(result)))

        # 4、将预测结果写入文件
        # 写入预测结果：output_dir/test_results.txt
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            # start_time = time.time()
            total_time_per_step = 0
            # probs_array=[]
            logits_array = []
            tf.logging.info("***** Predict results *****")
            for i, (elapsed_time, prediction) in enumerate(wem_utils.timer(result)):
                if i >= num_actual_predict_examples:
                    continue
                total_time_per_step += elapsed_time
                output_line = " ".join(str(edit) for edit in prediction["predictions"] if edit > 0) + "\n"
                # logits = np.array(prediction["logits"])
                # logits_array.append(logits)
                writer.write(output_line)
                num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples
            tf.logging.info("Decoding time: {}".format(total_time_per_step))
        # 写入预测概率：output_dir/test_logits.npz
        # output_logits_file = os.path.join(FLAGS.output_dir, "test_logits.npz")
        # with tf.gfile.GFile(output_logits_file, "w") as writer:
        #     np.save(writer, np.array(logits_array))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
