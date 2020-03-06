#!/bin/bash

# ab - Apache HTTP server benchmarking tool
# http://httpd.apache.org/docs/2.0/programs/ab.html
# parameters:
# -n requests     Number of requests to perform
# -c concurrency  Number of multiple requests to make at a time
# -t timelimit    Seconds to max. to spend on benchmarking
# -p postfile     File containing data to POST. Remember also to set -T
# -T content-type Content-type header to use for POST/PUT data, eg.
#                    'application/x-www-form-urlencoded'
#                    Default is 'text/plain'
# -v verbosity    How much troubleshooting info to print
# -V              Print version number and exit

ab -n 4 -c 1 -p 'post.json' -T 'application/json' 'http://127.0.0.1:12023/writingGrammarCorrect'
