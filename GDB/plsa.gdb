file bin/plsa.exe


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


define pPzdw
	pMatrix $arg0 NUM_TOPICS 1
end

define pPwz
	pMatrix $arg0 NUM_TOPICS NUM_VOCABS
end

define PSumPwz
	pMatrix $arg0 NUM_TOPICS 1
end

define pPzd
	set $num_docs = $arg0.size() / NUM_TOPICS
	printf "num docs : %d\n", $num_docs
	pMatrix $arg0 $num_docs NUM_TOPICS
end


#break main
#run Datasets/farsnews_corpus.txt 2

#break PLSA.cpp:318
#break PLSA.cpp:136
#break PLSA.cpp:150
#break PLSA.cpp:169
#run Datasets/small-corpus.txt 2

#break PLSA.cpp:367
#break PLSA.cpp:125
#break PLSA.cpp:134
break PLSA.cpp:147
run Datasets/train.txt Datasets/test.txt 2
#layout src

