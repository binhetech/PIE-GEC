#!/usr/bin/env python
# coding=utf-8
"""
本模块用于作文自动拼写、语法检查及纠错.

本模块中包含以下类:
    FlaskApp: 作文自动拼写、语法检查及纠错

本模块中包含以下方法:
    writingGrammarCorrect: 作文自动拼写、语法检查及纠错接口方法

本模块中包含以下属性:
    app: 作文自动拼写、语法检查及纠错应用程序
"""

from gevent import monkey

monkey.patch_all()

import os
import time
import json
import logging
import logging.handlers
from flask import Flask, jsonify, request
from pie import PieModel


class FlaskApp(Flask):
    """
    自动拼写、语法纠错应用程序.

    本类中包含以下方法:
        __init__: 初始化方法
        correct:  进行自动拼写、语法纠错
        logging:  进行日志记录

    本类中包含以下属性:
        None
    """

    def __init__(self, *args, **kwargs):
        """
        初始化方法.

        Args:
            None

        Return:
            None

        """
        super(FlaskApp, self).__init__(*args, **kwargs)
        self.corrector = PieModel(vocab_file="./resources/configs/vocab.txt",
                                  bert_config_file="./resources/configs/bert_config.json",
                                  path_inserts="./resources/models/conll/common_inserts.p",
                                  path_deletes="./resources/models/conll/common_deletes.p",
                                  path_multitoken_inserts="./resources/models/conll/common_multitoken_inserts.p",
                                  predict_checkpoint="./resources/models/pie_model.ckpt",
                                  output_dir="./resources/models", max_seq_length=128, use_tpu=False, inferMode="conll")
        # 获取日志存储工具
        self.logger = logging.getLogger('algorithm_nlp_basic_writing_grammarCorrect')
        self.logger.setLevel(logging.INFO)
        logDir = "./log/algorithm_nlp_basic_writing_grammarCorrect"
        if not os.path.exists(logDir):
            os.makedirs(logDir)
        # 设置日志回滚周期为30天
        fileName = os.path.join(logDir, "log")
        handler = logging.handlers.TimedRotatingFileHandler(filename=fileName,
                                                            when="D",
                                                            interval=1,
                                                            backupCount=30)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # 该进程处理的订单数量
        self.orderCount = 0

    def correct(self, text, lang="en", mode="mixed"):
        """
        自动拼写、语法纠错.

        Args:
            text: 输入文本
            lang: string, 语言类型, 默认"en"
            mode: string, 纠错模式，包括："spell_simple"-仅简单拼写纠错; "spell_lm"-仅语言模型拼写纠错;
                                        "grammar"-仅语法纠错；"mixed"-先语法后拼写同时纠错（默认值）
        Return:
            result: 字典. 返回结果

        """
        result = self.corrector.correct(text, lang, mode)
        return result

    def logging(self, appName, request, response, requestTime, responseTime, status):
        """
        用于生成关于服务的日志信息.

        Args:
            appName: str, 应用名称
            request: Request, 请求信息
            response: Response, 返回信息
            requestTime: float, 请求发生时间
            responseTime: float, 请求处理完毕时间
            status: str, 日志等级

        Return:
            None

        """
        log = {}
        log["appName"] = appName
        log["version"] = "1.0.0"
        localTime = time.localtime(time.time() + 28800)  # 日志生成时间戳
        log["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", localTime)

        # 生成request ID
        rt = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(requestTime + 28800))
        pid = os.getpid()
        self.orderCount += 1
        requestId = "%s.%d.%d" % (rt, pid, self.orderCount)

        log["requestId"] = requestId
        log["tags"] = []
        log["method"] = request.method.lower()
        log["level"] = status
        log["pid"] = pid
        log["statusCode"] = response.status_code
        log["errorCode"] = response.json["code"]

        log["req"] = {}
        log["req"]["url"] = request.url
        log["req"]["method"] = request.method.lower()
        headers = {k.lower(): v for k, v in dict(request.headers).items()}
        log["headers"] = headers
        log["req"]["remoteAddress"] = request.remote_addr
        log["req"]["userAgent"] = request.user_agent.string
        log["req"]["referrer"] = request.referrer
        log["req"]["contentLength"] = request.content_length
        log["req"]["content"] = json.loads(request.data)

        log["res"] = {}
        log["res"]["statusCode"] = response.status_code

        # 计算响应时间
        responseTime = int((responseTime - requestTime) * 1000)
        log["res"]["responseTime"] = responseTime
        log["res"]["contentLength"] = response.content_length
        log["res"]["content"] = response.json

        # 简要日志信息
        message = "%s: %s-%s %d %dms - code:%d" % (log["level"], log["appName"],
                                                   log["version"], log["statusCode"],
                                                   responseTime, log["errorCode"])
        log["message"] = message
        json.dumps(log)

        self.logger.info(json.dumps(log))


app = FlaskApp(__name__)


@app.route("/writingGrammarCorrect", methods=["POST"])
def writingGrammarCorrect():
    """
    作文自动拼写、语法纠错接口方法.

    Args:
        request: Request, 即请求信息

    Return:
        response: Response, 即响应信息
        response.status_code: int, HTTP状态码
        contentType: dict, 响应格式, 这里均为"application/json"

    """
    if request.method == "POST":
        appName = "algorithm_nlp_basic_writing_grammarCorrect"
        requestTime = time.time()
        form = json.loads(request.data)

        response = ""
        status = "info"  # 代码运行状态, 等于当前日志等级

        # 若表单为空, 返回错误信息日志
        if not form:
            status = "error"
            error = {"message": "Error: the request form is empty!",
                     "code": -1}
            response = jsonify(error)

        # 获取输入文本
        text = ""
        if status == "info":
            if "text" in form:
                text = form["text"]
            else:
                status = "error"
                error = {"message": "Error: the request doesn't have key 'content'!",
                         "code": -1}
                response = jsonify(error)

        # 获取输入语言类型
        lang = "en"
        if status == "info":
            if "lang" in form:
                lang = form["lang"]

        # 获取纠错模式
        mode = "mixed"
        if status == "info":
            if "mode" in form:
                mode = form["mode"]

        if status == "info":
            try:
                result = app.correct(text, lang, mode)
                if result["code"] < 0:
                    # warning with info in "message"
                    status = "warning"
                    result["code"] = 0
                response = jsonify(result)
            except KeyError as e:
                error = {"message": "Error: can not find the key %s! " % (str(e)),
                         "code": -1}
                response = jsonify(error)
            except Exception as e:
                status = "error"
                error = {"message": f"Error: {type(e).__name__} -- {str(e)}!",
                         "code": -2}
                response = jsonify(error)

        responseTime = time.time()

        # 将返回结果按照指定日志格式输出
        app.logging(appName, request, response, requestTime, responseTime, status)
        return response
    else:
        return jsonify({"message": "Error: please use POST method!", "code": -4, })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12023)
