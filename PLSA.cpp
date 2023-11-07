#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define PW_Z_IDX(W, K)			((K * NUM_VOCABS) + W)
#define PZ_D_IDX(Z, D)			((D * NUM_TOPICS) + Z)

#define IS_INF_OR_NAN(F)	(std::isinf(F) || std::isinf(-F) || (F != F))



// Constants
constexpr int MAX_TRAINING_ITERATIONS = 100;
constexpr int MAX_TEST_ITERATIONS = 25;
constexpr int MIN_ITERATIONS = 10;
constexpr double EPSILON = 5.0;

/// Parameters
int NUM_TOPICS = 0;					/// K
int NUM_VOCABS = 0;					/// V
int MAX_WORDS_IN_DOC = 0;			/// N
int MIN_TERM = 0;



struct Word
{
	int term;
	int count;
};

typedef std::vector<Word> Doc;
typedef std::vector<Doc> Corpus;



bool ReadCorpus(const char* corpus_path, Corpus& corpus, std::vector<int>& doc_count,
		int& out_min_term, int& out_num_vocabs, int& out_max_words_in_doc)
{
	std::ifstream in_file(corpus_path);
	if (!in_file.is_open())
		return false;

	int min_term = std::numeric_limits<int>::max();
	int max_term = std::numeric_limits<int>::min();
	int max_words_in_doc = std::numeric_limits<int>::min();
	corpus.clear();
	doc_count.clear();
	std::string line;
	while (std::getline(in_file, line)) {
		std::istringstream iss(line);
		int num_terms;
		iss >> num_terms;													// Read number of words in current document.

		// Read words.
		Doc doc;
		int total_count = 0;
		for (int n = 0; n < num_terms; ++n) {
			char colon;
			Word word;
			iss >> word.term >> colon >> word.count;						// Read term index and count.

			min_term = std::min(min_term, word.term);
			max_term = std::max(max_term, word.term);
			total_count += word.count;
			doc.push_back(word);
		}
		max_words_in_doc = std::max(max_words_in_doc, total_count);
		corpus.push_back(doc);
		doc_count.push_back(total_count);
	}

	out_min_term = min_term;
	out_num_vocabs = max_term - min_term + 1;
	out_max_words_in_doc = max_words_in_doc;
	return true;
}

double CalcLogLikelihood(const Corpus& corpus, const std::vector<int>& doc_count,
		const std::vector<double>& Pw_z, const std::vector<double>& Pz_d)
{
	double sum_n_d = std::accumulate(doc_count.begin(), doc_count.end(), 0);
	double likelihood = 0.0;
	for (int i = 0; i < static_cast<int>(corpus.size()); ++i) {
		const Doc& doc = corpus[i];
		for (int j = 0; j < static_cast<int>(doc.size()); ++j) {
			const int w_j = doc[j].term - MIN_TERM;
			double sum_Pw_z_Pz_d = 0.0;
			for (int k = 0; k < NUM_TOPICS; ++k)
				sum_Pw_z_Pz_d += Pw_z[PW_Z_IDX(w_j, k)] * Pz_d[PZ_D_IDX(k, i)];
			likelihood += doc[j].count * log(sum_Pw_z_Pz_d);
		}
		double Pd = log(doc_count[i]) - log(sum_n_d);
		likelihood += doc_count[i] * Pd;
	}
	return likelihood;
}

double RunEM(int max_iterations, const Corpus& corpus, const std::vector<int>& doc_count,
		std::vector<double>& Pw_z, std::vector<double>& Pz_d)
{
	double log_likelihood = 0.0;
	double old_likelihood = -std::numeric_limits<double>::max();
	std::vector<double> Pz_dw(NUM_TOPICS);									// Temporary space for storing P(z_k | d_i, w_j)
	std::vector<double> tmp_Pw_z(NUM_VOCABS * NUM_TOPICS);					// \sum_{i=1}^N n(d_i, w_j) P(z_k | d_i, W_j)
	std::vector<double> sum_Pw_z(NUM_TOPICS);								// \sum_{m=1}^M \sum_{i=1}^N n(d_i, w_m) * P(z_k | d_i, w_m)
	std::vector<double> tmp_Pz_d(NUM_TOPICS * corpus.size());

	for (int itr = 0; itr < max_iterations; ++itr) {
		std::cout << " ---------- itr #" << (itr + 1) << " (of " << max_iterations << ") ----------" << std::endl;

		// E-step
		for (int i = 0; i < static_cast<int>(corpus.size()); ++ i) {		// Documents
			const Doc& doc = corpus[i];
			for (int j = 0; j < static_cast<int>(doc.size()); ++j) {		// Words
				const int w_j = doc[j].term - MIN_TERM;
				double sum_Pz_dw = 0.0;										// \sum_k P(z_k | d_i, w_j)    Normalizer constant

				// Calculate P(z | d, w)
				for (int k = 0; k < NUM_TOPICS; ++k) {						// Topics
					double p = Pw_z[PW_Z_IDX(w_j, k)] * Pz_d[PZ_D_IDX(k, i)];
					if (p == 0) {
						//std::cout << "zero p at k:" << k << " i:" << i << " j:" << j << std::endl;
						p = 1e-10;
					}
					Pz_dw[k] = p;
					sum_Pz_dw += p;
				}

				for (int k = 0; k < NUM_TOPICS; ++k) {
					//if (sum_Pz_dw == 0)
					//	std::cout << "zero sum at k:" << k << " i:" << i << " j:" << j << std::endl;
					double p = Pz_dw[k] / sum_Pz_dw;						// Normalize P(z | d, w).
					if (p == 0) {
						//std::cout << "zero2 p at k:" << k << " i:" << i << " j:" << j << std::endl;
						p = 1e-10;
					}

					const double n_p = doc[j].count * p;
					tmp_Pw_z[PW_Z_IDX(w_j, k)] += n_p;
					sum_Pw_z[k] += n_p;

					tmp_Pz_d[PZ_D_IDX(k, i)] += n_p;
				}
			}		// j
		}		// i

		// M-step
		for (int k = 0; k < NUM_TOPICS; ++k) {
			// Normalize P(w | z)
			for (int j = 0; j < NUM_VOCABS; ++j) {
				const int PW_ZIDX = PW_Z_IDX(j, k);
				if (tmp_Pw_z[PW_ZIDX] == 0) {
					//std::cout << "tmp_Pw_z[PW_ZIDX:" << PW_ZIDX << "]:" << tmp_Pw_z[PW_ZIDX] << "   nan or zero! k:" << k << " j:" << j << std::endl;
					tmp_Pw_z[PW_ZIDX] = 10e-10;
				}
				Pw_z[PW_ZIDX] = tmp_Pw_z[PW_ZIDX] / sum_Pw_z[k];
				tmp_Pw_z[PW_ZIDX] = 0;
			}
			sum_Pw_z[k] = 0;

			// Normalize P(z | d)
			for (int i = 0; i < static_cast<int>(corpus.size()); ++i) {
				const int PZ_DIDX = PZ_D_IDX(k, i);
				Pz_d[PZ_DIDX] = tmp_Pz_d[PZ_DIDX] / doc_count[i];
				tmp_Pz_d[PZ_DIDX] = 0;
			}
		}

		// Calculate log likelihood.
		log_likelihood = CalcLogLikelihood(corpus, doc_count, Pw_z, Pz_d);
		std::cout << " Log likelihood:  " << static_cast<int>(log_likelihood) << std::endl;
		std::cout << " old likelihood:  " << static_cast<int>(old_likelihood) << std::endl;

		// Check convergence.
		double diff_likelihood = log_likelihood - old_likelihood;
		std::cout << " diff likelihood: " << diff_likelihood << std::endl;
		if (diff_likelihood < EPSILON && itr >= MIN_ITERATIONS) {
			std::cout << "********** Converged! **********" << std::endl;
			break;
		}

		old_likelihood = log_likelihood;
	}

	return log_likelihood;
}

void DumpPw_z(const std::vector<double>& Pw_z, const char* out_path = "Pw_z.txt")
{
	std::ofstream out_file(out_path);
	out_file.precision(7);
	out_file << std::fixed;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		out_file << k << " ->   " << Pw_z[PW_Z_IDX(0, k)];
		for (int j = 1; j < NUM_VOCABS; ++j)
			out_file << ' ' << Pw_z[PW_Z_IDX(j, k)];
		if (k + 1 < NUM_TOPICS)
			out_file << std::endl;
	}
}

int GetDocTopic(int i, const std::vector<double>& Pz_d)
{
	int mk = 0;
	double mz = Pz_d[PZ_D_IDX(0, i)];
	for (int k = 1; k < NUM_TOPICS; ++k)
		if (Pz_d[PZ_D_IDX(k, i)] > mz) {
			mk = k;
			mz = Pz_d[PZ_D_IDX(k, i)];
		}
	return mk;
}

void DumpPz_d(const std::vector<double>& Pz_d, const char* out_path = "Pz_d.txt")
{
	std::ofstream out_file(out_path);
	out_file.precision(7);
	out_file << std::fixed;
	const int num_docs = static_cast<int>(Pz_d.size()) / NUM_TOPICS;
	for (int i = 0; i < num_docs; ++i) {
		int doc_topic = GetDocTopic(i, Pz_d);
		out_file << std::setw(3) << i << " ->  " << doc_topic << "    " << Pz_d[PZ_D_IDX(0, i)];
		for (int k = 1; k < NUM_TOPICS; ++k)
			out_file << "   " << Pz_d[PZ_D_IDX(k, i)];
		if (i + 1 < num_docs)
			out_file << std::endl;
	}
}

void CalcAccuracy(const std::vector<double>& Pz_d)
{
	if (NUM_TOPICS > 5)
		return;

	std::vector<std::vector<int>> cluster_count(NUM_TOPICS, std::vector<int>(NUM_TOPICS, 0));

	// Count clusters.
	const int num_docs = static_cast<int>(Pz_d.size() / NUM_TOPICS);
	const int NUM_DOCS_IN_TOPIC = num_docs / NUM_TOPICS;
	for (int k = 0; k < NUM_TOPICS; ++k)
		for (int d = 0; d < NUM_DOCS_IN_TOPIC; ++d)
			++cluster_count[k][GetDocTopic(k * NUM_DOCS_IN_TOPIC + d, Pz_d)];

	std::vector<int> perm(NUM_TOPICS);
	for (int i = 0; i < NUM_TOPICS; ++i)
		perm[i] = i;

	// Calculate accuracies and find maximum one for report.
	std::vector<double> accs;
	do {
		double sm = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k)
			sm += cluster_count[k][perm[k]];
		const double ACC = sm / static_cast<double>(num_docs);
		accs.push_back(ACC);
	} while (std::next_permutation(perm.begin(), perm.end()));

	// Print resul.
	std::cout << " Accuracy: " << *std::max_element(accs.begin(), accs.end()) << "   [ ";
	for (const auto& a : accs)
		std::cout << a << ' ';
	std::cout << ']' << std::endl;
}



int main(int argc, char** argv)
{
	// Check number of command line arguments.
	if (argc <= 3) {
		// Print usage.
		std::cout << "Usage:   " << argv[0] << " CORPUS_PATH  TEST_PATH  K" << std::endl;
		return 1;
	}

	// Parse command line arguments.
	const char* input_file = argv[1];
	const char* test_file = argv[2];
	NUM_TOPICS = atoi(argv[3]);

	Corpus corpus;															// Training dataset
	std::vector<int> doc_count;												// n(d_i) Number of words in each document
	if (!ReadCorpus(input_file, corpus, doc_count, MIN_TERM, NUM_VOCABS, MAX_WORDS_IN_DOC)) {
		std::cout << "Could not read `" << input_file << "' file!" << std::endl;
		return 1;
	}

	Corpus test_corpus;														// Test dataset (held out documents)
	std::vector<int> test_count;											// n'(d_i) Number of words in each test document
	int test_min_term, test_num_vocabs, test_max_word_in_doc;
	if (!ReadCorpus(test_file, test_corpus, test_count, test_min_term,
				test_num_vocabs, test_max_word_in_doc)) {
		std::cout << "Could not read `" << test_file << "' test file!" << std::endl;
		return 1;
	}

	// Update some parameters.
	int max_term = std::max(MIN_TERM + NUM_VOCABS, test_min_term + test_num_vocabs);
	MIN_TERM = std::min(MIN_TERM, test_min_term);
	NUM_VOCABS = max_term - MIN_TERM;										// The max_term equals to max term index + 1 now, so we don't need to subtract one.
	MAX_WORDS_IN_DOC = std::max(MAX_WORDS_IN_DOC, test_max_word_in_doc);

	// Print some statistics.
	std::cout << "Num topics (K)       : " << NUM_TOPICS << std::endl;
	std::cout << "Num vocabularies (V) : " << NUM_VOCABS << std::endl;
	std::cout << "Corpus size (M)      : " << corpus.size() << std::endl;
	std::cout << "Test size (q)        : " << test_corpus.size() << std::endl;
	std::cout << "Max num words in doc : " << MAX_WORDS_IN_DOC << std::endl;

	// Initialize model parameters by random values.
	std::vector<double> Pw_z;												// P(w | z)   it is equivalent to \beta in LDA context
	Pw_z.resize(NUM_VOCABS * NUM_TOPICS, 1.0 / NUM_VOCABS);
	std::random_device d;
	std::uniform_real_distribution<double> u(0.0, 1.0);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		double sum = 0.0;
		for (int j = 0; j < static_cast<int>(Pw_z.size() / NUM_TOPICS); ++j) {
			Pw_z[PW_Z_IDX(j, k)] += u(d);
			sum += Pw_z[PW_Z_IDX(j, k)];
		}
		for (int j = 0; j < static_cast<int>(Pw_z.size() / NUM_TOPICS); ++j)
			Pw_z[PW_Z_IDX(j, k)] /= sum;
	}

	std::vector<double> Pz_d;												// P(z | d)   it is equivalent to \gamma in LDA context
	Pz_d.resize(NUM_TOPICS * corpus.size(), 1.0 / NUM_TOPICS);
	for (int i = 0; i < static_cast<int>(Pz_d.size() / NUM_TOPICS); ++i) {
		double sum = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k) {
			Pz_d[PZ_D_IDX(k, i)] += u(d);
			sum += Pz_d[PZ_D_IDX(k, i)];
		}
		for (int k = 0; k < NUM_TOPICS; ++k)
			Pz_d[PZ_D_IDX(k, i)] /= sum;
	}

	std::vector<double> Pz_q;												// P(z | q)   q stands for QUERY don't ask me why :)
	Pz_q.resize(NUM_TOPICS * test_corpus.size(), 1.0 / NUM_TOPICS);
	for (int i = 0; i < static_cast<int>(Pz_q.size() / NUM_TOPICS); ++i) {
		double qsum = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k) {
			Pz_q[PZ_D_IDX(k, i)] += u(d);
			qsum += Pz_q[PZ_D_IDX(k, i)];
		}
		for (int k = 0; k < NUM_TOPICS; ++k)
			Pz_q[PZ_D_IDX(k, i)] /= qsum;
	}

	std::cout << "===== R U N   T R A I N I N G   E M =====" << std::endl;
	RunEM(MAX_TRAINING_ITERATIONS, corpus, doc_count, Pw_z, Pz_d);			// Run EM algorithm.
	DumpPw_z(Pw_z);															// Write results and calculate accuracy for training set.
	DumpPz_d(Pz_d);
	CalcAccuracy(Pz_d);

	std::cout << std::endl << std::endl;
	std::cout << "===== R U N   T E S T   E M =====" << std::endl;
	double log_likelihood = RunEM(MAX_TEST_ITERATIONS, test_corpus, test_count, Pw_z, Pz_q);		// Run EM with folding in test data.
	DumpPw_z(Pw_z, "Pw_q.txt");												// Write results and calculate accuracy for test set.
	DumpPz_d(Pz_q, "Pz_q.txt");
	std::cout << "K: " << NUM_TOPICS << std::endl;
	std::cout << "log likelihood: " << static_cast<int>(log_likelihood) << std::endl;
	CalcAccuracy(Pz_q);
	return 0;
}

