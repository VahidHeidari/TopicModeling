#!/bin/bash

FILE_NAME=LDA-`date +%a_%Y-%m-%d`
tar -czf $FILE_NAME.tar.gz CMakeLists.txt small-corpus.txt big-corpus.txt	\
	archive.sh build.sh LDA.cpp

