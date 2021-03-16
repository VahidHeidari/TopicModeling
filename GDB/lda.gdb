file bin/inf.exe


define pCorpus
	printf "corpus size: %d\n", $arg0.size()
	set $i = 0
	while $i < $arg0.size()
		printf "corpus[%d] ---- ", $i + 1
		pDoc $arg0[$i]
		set $i = $i + 1
	end
end


define pDoc
	printf "doc size: %d  ->  ", $arg0.size()
	set $n = 0
	while $n < $arg0.size()
		printf " %d:%d ", $arg0[$n].term, $arg0[$n].count
		set $n = $n + 1
	end
	printf "\n"
end


#
# arg0 : The std::vector<double> matrix to print
# arg1 : Number of rows
# arg2 : Number of columns
#
define pMatrix
	set $NROW = $arg1
	set $NCOL = $arg2
	set $y = 0
	while $y < $NROW
		printf "y %d  -> ", $y
		set $x = 0
		while $x < $NCOL
			set $idx = ($y * $NCOL) + $x
			printf " %2.2f ", $arg0._M_impl._M_start[$idx]
			set $x = $x + 1
		end
		printf "\n"
		set $y = $y + 1
	end
end


define pBeta
	pMatrix log_beta NUM_TOPICS NUM_VOCABS
end

define pTmpBeta
	pMatrix tmp_beta NUM_TOPICS NUM_VOCABS
end

define pSumBeta
	pMatrix sum_beta NUM_TOPICS 1
end

define pGamma
	pMatrix var_gamma NUM_DOCS NUM_TOPICS
end

define pTGamma
	pMatrix var_gamma TEST_NUM_DOCS NUM_TOPICS
end


define pPhi
	set $d = 0
	while $d < NUM_DOCS
		printf "--- doc[%d] ----\n", $d
		set $n = 0
		while $n < NUM_VOCABS
			set $k = 0
			while $k < NUM_TOPICS
				set $idx = ($d * NUM_VOCABS * NUM_TOPICS) + ($n * NUM_TOPICS) + $k
				printf " %2.2f ", phi._M_impl._M_start[$idx]
				set $k = $k + 1
			end
			printf "\n"
			set $n = $n + 1
		end
		set $d = $d + 1
	end
end


define pTPhi
	set $d = 0
	while $d < TEST_NUM_DOCS
		printf "--- doc[%d] ----\n", $d
		set $n = 0
		while $n < NUM_VOCABS
			set $k = 0
			while $k < NUM_TOPICS
				set $idx = ($d * NUM_VOCABS * NUM_TOPICS) + ($n * NUM_TOPICS) + $k
				printf " %2.2f ", phi._M_impl._M_start[$idx]
				set $k = $k + 1
			end
			printf "\n"
			set $n = $n + 1
		end
		set $d = $d + 1
	end
end


#break LDA.cpp:97
#break LDA.cpp:231
break inference.cpp:404
break inference.cpp:418 if d == 102
#break inference.cpp:429
run Datasets/farsnews_corpus.txt Datasets/test_corpu.txt 2
layout src

