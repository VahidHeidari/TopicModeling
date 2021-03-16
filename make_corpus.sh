#!/bin/bash

python Python/get_pages.py													\
	&& python Python/collect_vocabularies.py								\
	&& python Python/make_corpus_and_test_data.py 900

