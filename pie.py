#!/usr/bin/env python
# coding=utf-8

"""
本模块是利用Parallel Iterative Edit Models进行自动语法纠错.

本模块中包含以下类:
    PieModel: Parallel Iterative Edit Models类

本模块中包含以下方法:
    None

本模块中包含以下属性:


"""
import sys

sys.path.append("./word_edit")

import pickle
import tensorflow as tf
from tensorflow.contrib.distribute import AllReduceCrossDeviceOps
from nltk import sent_tokenize

from word_edit.wem_utils import list_to_ids
from word_edit.modeling import BertConfig
from word_edit.word_edit_model import FLAGS, sequence_padding, gec_model_fn_builder
from word_edit.tokenization import FullTokenizer
from word_edit.tokenize_input import get_tuple
from word_edit.opcodes import Opcodes
from word_edit.apply_opcode import apply_opcodes


class PieModel(object):
    """
    Parallel Iterative Edit Models进行自动语法纠错类.

    本类中包含以下方法:
        __init__: 初始化方法
        predict_sentences: 针对句子列表进行纠错预测
        correct: 针对输入错误的文本进行语法纠错

    本类中包含以下属性:
        None

    """

    def __init__(self, vocab_file="./resources/configs/vocab.txt",
                 bert_config_file="./resources/configs/bert_config.json",
                 path_inserts="./resources/models/conll/common_inserts.p",
                 path_deletes="./resources/models/conll/common_deletes.p",
                 path_multitoken_inserts="./resources/models/conll/common_multitoken_inserts.p",
                 predict_checkpoint="./resources/models/pie_model.ckpt",
                 output_dir="./resources/models", max_seq_length=128, use_tpu=False, numGpu=2, inferMode="conll",
                 doSpellCheck=True):
        """
        初始化方法.

        Args:
            vocab_file: string, bert的词典文件路径
            bert_config_file: string, bert的配置文件路径
            path_inserts: string, 插入token文件路径
            path_deletes: string, 删除token文件路径
            path_multitoken_inserts: string, 多token插入文件路径
            predict_checkpoint: string, 预测模型文件路径
            max_seq_length: int, 最大序列长度
            use_tpu: boolean, 是否使用tpu，默认不使用tpu, 而是使用GPU运算(当无gpu设备时切换至cpu运算)
            numGpu: int, GPU使用个数，仅当开启gpu运算且GPU可用的情况下使用
            inferMode: string, 推理模式，包括："conll", "bea"
            doSpellCheck: boolean, 是否先进行简单拼写检查

        Return:
            None

        """
        FLAGS.do_train = False
        FLAGS.do_eval = False
        FLAGS.do_predict = True
        FLAGS.do_lower_case = False
        FLAGS.bert_config_file = bert_config_file
        FLAGS.vocab_file = vocab_file
        FLAGS.path_inserts = path_inserts
        FLAGS.path_multitoken_inserts = path_multitoken_inserts
        FLAGS.predict_checkpoint = predict_checkpoint
        FLAGS.output_dir = output_dir
        FLAGS.max_seq_length = max_seq_length
        FLAGS.use_tpu = use_tpu
        FLAGS.n_gpus = numGpu
        FLAGS.init_checkpoint = False
        FLAGS.predict_batch_size = 128
        self.use_tpu = use_tpu
        self.max_seq_length = max_seq_length
        self.do_spell_check = doSpellCheck
        self.inferMode = inferMode
        # 构建tokenizer对象
        self.tokenizer = FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        # 编辑操作代码
        self.opcodes = Opcodes(path_inserts, path_deletes, path_multitoken_inserts, True)
        # 构建估计器对象
        self.estimator = self._loading_estimator(FLAGS)
        self.idsList = []
        self.first_run = True
        self.predictions = None
        self.closed = False
        # 第一次运行检查
        self._first_check()
        return

    def _loading_estimator(self, FLAGS):
        """
        初始化及加载估计器.

        Args:
            FLAGS: class, 命令行参数解析类

        Return:

        """
        if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
            raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")
        # 使用tpu时需要drop out剩余不足一个batch size的样本
        self.predict_drop_remainder = True if FLAGS.use_tpu else False
        # bert配置文件
        bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
        if FLAGS.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" % (
                    FLAGS.max_seq_length, bert_config.max_position_embeddings))
        # 读取：插入编辑tokens、id
        inserts = pickle.load(tf.gfile.Open(FLAGS.path_inserts, "rb"))
        insert_ids = self.tokenizer.convert_tokens_to_ids(inserts)
        # 读取：多token插入编辑tokens、id
        multitoken_inserts = pickle.load(tf.gfile.Open(FLAGS.path_multitoken_inserts, "rb"))
        multitoken_insert_ids = list_to_ids(multitoken_inserts, self.tokenizer)

        num_train_steps = None
        num_warmup_steps = None

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

        # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
        if FLAGS.use_tpu:
            # 1.创建tpu集群
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name,
                                                                                  zone=FLAGS.tpu_zone,
                                                                                  project=FLAGS.gcp_project)
            # 2.设置运行配置run_config
            is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
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
            # 3.构造TPU估计器对象，用于TPU训练
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
                params={"batch_size": FLAGS.predict_batch_size})
        return estimator

    def _word_tokenize(self, sentences, doSpellCheck=True):
        """
        针对句子列表进行word piece级tokenize.

        Args:
            sentences: list of string, 句子列表
            doSpellCheck: boolean, 是否先进行简单拼写检查

        Return:
            tokensList：list of list of string
            idsList： list of list of int

        """
        tokensList, idsList = [], []
        for sentence in sentences:
            # 针对句子进行word piece tokenize
            tokens, token_ids = get_tuple(self.tokenizer, sentence, doSpellCheck)
            tokensList.append(tokens)
            idsList.append(token_ids)
        return tokensList, idsList

    def _get_feature(self, example):
        """提取样本的特征数据."""
        return sequence_padding(example, None, self.max_seq_length)

    def _create_generator(self):
        """构建数据输入的生成器."""
        while not self.closed:
            features = (self._get_feature(f) for f in self.idsList)
            yield dict(zip(("input_sequence", "input_mask", "segment_ids", "edit_sequence"), zip(*features)))

    def _input_fn_builder(self):
        """构建数据输入函数."""
        dataset = tf.data.Dataset.from_generator(self._create_generator,
                                                 output_types={'input_sequence': tf.int64,
                                                               'input_mask': tf.int64,
                                                               'segment_ids': tf.int64,
                                                               'edit_sequence': tf.int64},
                                                 output_shapes={'input_sequence': (None, self.max_seq_length),
                                                                'input_mask': (None, self.max_seq_length),
                                                                'segment_ids': (None, self.max_seq_length),
                                                                'edit_sequence': (None, self.max_seq_length)}
                                                 )
        iterator = dataset.make_one_shot_iterator()
        dataset = iterator.get_next()
        return dataset

    def _clean_up_tokenization(self, text):
        """Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms."""
        out_string = (
            text.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" do not", " don't")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
        )
        return out_string

    def predict_sentences(self, sentences, doSpellCheck=True, doCleanUp=True):
        """
        针对句子列表进行纠错预测.

        Args:
            sentences: list of string, 句子列表
            doSpellCheck: boolean, 是否先进行autocorrect拼写检查
            doCleanUp: boolean, 是否进行空格及特殊字符的缩进处理

        Return:
            corSentences: list of string, 纠错后的句子列表

        """
        if doSpellCheck is None:
            doSpellCheck = self.do_spell_check
        # 对句子列表进行token化
        tokensList, self.idsList = self._word_tokenize(sentences, doSpellCheck)
        # 构造输入函数
        if self.first_run:
            self.predictions = self.estimator.predict(input_fn=self._input_fn_builder,
                                                      checkpoint_path=FLAGS.predict_checkpoint,
                                                      yield_single_examples=False)
            self.first_run = False
        # 解析纠错结果
        corSentences = []
        # next generator
        prediction = next(self.predictions)
        for i in range(len(sentences)):
            editList = [edit for edit in prediction["predictions"][i] if edit > 0]
            # 应用编辑操作进行纠正
            corTokens = apply_opcodes(tokensList[i], editList, self.opcodes, self.tokenizer.basic_tokenizer,
                                      self.inferMode)
            if doCleanUp:
                corSent = self._clean_up_tokenization(" ".join(corTokens))
            else:
                corSent = " ".join(corTokens)
            corSentences.append(corSent)
        return corSentences

    def correct(self, text, lang="en", mode="grammar", hyphen=" ", doSentToken=True, doSpellCheck=None, numRound=4,
                doCleanUp=True):
        """
        针对输入错误的文本进行语法纠错.

        Args:
            text: string, 输入的纠错文本
            lang: string, 输入文本的语言，目前只支持"en"
            mode: string, 纠错模式，目前无用
            hyphen: string, 句子链接字符，默认为空格
            doSentToken: boolean, 是否先对输入文件进行分句，默认true
            doSpellCheck: boolean, 是否先进行简单拼写检查
            numRound: int, 迭代编辑次数
            doCleanUp: boolean, 是否进行空格及特殊字符的缩进处理

        Return:
            result: dict

        """
        if doSentToken:
            oriSentences = sent_tokenize(text)
        else:
            oriSentences = [text]
        # multi round
        corSentences = oriSentences.copy()
        for i in range(numRound):
            if i > 0:
                # 只在第一次进行拼写检查
                doSpellCheck = False
            corSentences = self.predict_sentences(corSentences, doSpellCheck, doCleanUp)
        # 构建结果输出
        result = {"text": text, "correction": hyphen.join(corSentences), "sentences": oriSentences,
                  "corSent": corSentences, "code": 0, }
        return result

    def _first_check(self):
        """第一次检查句子以加载模型常驻内存."""
        self.correct("Jack is the writer who have liveed in the beijing citys.")
        return
