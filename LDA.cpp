#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define PHI_IDX(D, N, K) ((D * NUM_VOCABS * NUM_TOPICS) + (N * NUM_TOPICS) + K)
#define GAMMA_IDX(D, K)  ((D * NUM_TOPICS) + K)
#define BETA_IDX(K, N)   ((K * NUM_VOCABS) + N)



// Constants
constexpr int MAX_ITERATIONS = 100;
constexpr int MIN_ITERATIONS = 2;
constexpr double EPSILON = 5.0;

/// Parameters
constexpr double ALPHA = 0.5;

int NUM_TOPICS = 2;					/// K
int NUM_VOCABS = 0;					/// V
int NUM_WORDS_IN_DOC = 0;			/// N
int NUM_DOCS = 0;					/// D

std::vector<double> var_gamma;		/// [D][K]
std::vector<double> phi;			/// [D][N][K]
std::vector<double> log_beta;		/// [K][V]
std::vector<double> tmp_beta;		/// [K][V]
std::vector<double> sum_beta;		/// [K]



struct Word
{
	int term;
	int count;
};

typedef std::vector<Word> Doc;
typedef std::vector<Doc> Corpus;



bool ReadCorpus(const char* corpus_path, Corpus& corpus)
{
	std::ifstream in_file(corpus_path);
	if (!in_file.is_open())
		return false;

	corpus.clear();
	std::string line;
	while (std::getline(in_file, line)) {
		// Read number of words in current document.
		size_t spos = line.find(' ', 0);
		int num_words = atoi(line.substr(0, spos).c_str());
		++spos;

		// Read words.
		Doc doc;
		for (int n = 0; n < num_words; ++n) {
			Word word;
			// Read word index.
			size_t epos = line.find(':', spos);
			word.term = atoi(line.substr(spos, epos - spos).c_str());
			spos = epos + 1;

			// Read word count.
			epos = line.find(' ', spos);
			if (epos == std::string::npos)
				epos = line.length();
			word.count = atoi(line.substr(spos, epos - spos).c_str());
			spos = epos + 1;

			doc.push_back(word);
		}
		corpus.push_back(doc);
	}
	return true;
}

int GetNumWords(const Doc& doc)
{
	int sum = 0;
	for (const auto& d : doc)
		sum += d.count;
	return sum;
}

void PrintBeta(const std::vector<double>& b)
{
	auto p = std::cout.precision();
	std::cout.precision(2);
	std::cout << std::fixed;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		std::cout << k << " ->   " << exp(b[BETA_IDX(k, 0)]);
		for (int n = 1; n < NUM_VOCABS; ++n)
			std::cout << ' ' << exp(b[BETA_IDX(k, n)]);
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout.precision(p);
}

int GetDocTopic(int d, const std::vector<double>& g)
{
	int mk = 0;
	double mg = g[GAMMA_IDX(d, 0)];
	for (int k = 1; k < NUM_TOPICS; ++k)
		if (g[GAMMA_IDX(d, k)] > mg) {
			mk = k;
			mg = g[GAMMA_IDX(d, k)];
		}
	return mk;
}

void PrintGamma(const std::vector<double>& g)
{
	for (int d = 0; d < NUM_DOCS; ++d) {
		int doc_topic = GetDocTopic(d, g);
		std::cout << std::setw(3) << d << " ->  " << doc_topic << "  " << g[GAMMA_IDX(d, 0)];
		for (int k = 1; k < NUM_TOPICS; ++k)
			std::cout << "   " << g[GAMMA_IDX(d, k)];
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

double digamma(double x)
{
	double p;
	x = x + 6;
	p = 1 / (x * x);
	p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333) * p - 0.083333333333333) * p;
	p = p + log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6);
	return p;
}

double CalcDocLikelihood(int d, const Corpus& corpus)
{
	double gamma_sum = 0;
	std::vector<double> digamma_gamma(NUM_TOPICS, 0.0);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		const int GIDX = GAMMA_IDX(d, k);
		digamma_gamma[k] = digamma(var_gamma[GIDX]);
		gamma_sum += var_gamma[GIDX];
	}
	double digamma_gamma_sum = digamma(gamma_sum);
	double likelihood = - NUM_TOPICS * lgamma(ALPHA) - lgamma(gamma_sum);
	const Doc& doc = corpus[d];
	for (int k = 0; k < NUM_TOPICS; ++k) {
		const double DIG = digamma_gamma[k] - digamma_gamma_sum;
		const int GIDX = GAMMA_IDX(d, k);
		likelihood += (ALPHA - 1) * DIG + lgamma(var_gamma[GIDX]) - (var_gamma[GIDX] - 1) * DIG;
		for (int n = 0; n < static_cast<int>(doc.size()); ++n) {
			const int PIDX = PHI_IDX(d, n, k);
			likelihood += phi[PIDX] * (DIG - log(phi[PIDX]) + (doc[n].count * log_beta[BETA_IDX(k, n)]));
		}
	}
	return likelihood;
}

double LogSum(double log_a, double log_b)
{
	if (log_b < log_a)
		std::swap(log_a, log_b);

	double res = log_b + log(1 + exp(log_a - log_b));
	return res;
}



int main(int argc, char** argv)
{
	const char* input_file = nullptr;

	// Parse command line arguments.
	if (argc > 2) {
		input_file = argv[1];
		NUM_TOPICS = atoi(argv[2]);
	} else {
		// Print usage.
		std::cout << "Usage:   " << argv[0] << " CORPUS_PATH   K" << std::endl;
		return 1;
	}

	Corpus corpus;
	if (!ReadCorpus(input_file, corpus)) {
		std::cout << "Could not read `" << input_file << "' file!" << std::endl;
		return 1;
	}

	// Update some parameters.
	NUM_DOCS = static_cast<int>(corpus.size());
	NUM_VOCABS = static_cast<int>(corpus[0].size());
	NUM_WORDS_IN_DOC = GetNumWords(corpus[0]);

	// Print some statistics.
	std::cout << "Num topics        : " << NUM_TOPICS << std::endl;
	std::cout << "Num vocabularies  : " << NUM_VOCABS << std::endl;
	std::cout << "Corpus size       : " << corpus.size() << std::endl;
	std::cout << "Num words per doc : " << NUM_WORDS_IN_DOC << std::endl;

	// Initialize parameters.
	var_gamma.resize(NUM_DOCS * NUM_TOPICS, ALPHA + (static_cast<double>(NUM_WORDS_IN_DOC) / NUM_TOPICS));
	phi.resize(NUM_DOCS * NUM_VOCABS * NUM_TOPICS, 1.0 / NUM_TOPICS);
	log_beta.resize(NUM_TOPICS * NUM_VOCABS, 0);
	tmp_beta.resize(NUM_TOPICS * NUM_VOCABS, 1.0 / NUM_TOPICS);
	sum_beta.resize(NUM_TOPICS, 0);
	std::random_device d;
	std::uniform_real_distribution<double> u(0.0, 1.0);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		for (int n = 0; n < NUM_VOCABS; ++n) {
			const int BIDX = BETA_IDX(k, n);
			tmp_beta[BIDX] += u(d);
			sum_beta[k] += tmp_beta[BIDX];
		}
		const double LOG_SUM_CLASS = log(sum_beta[k]);
		for (int n = 0; n < NUM_VOCABS; ++n) {
			const int BIDX = BETA_IDX(k, n);
			log_beta[BIDX] = log(tmp_beta[BIDX]) - LOG_SUM_CLASS;
		}
	}

	double old_likelihood = -std::numeric_limits<double>::min();
	for (int itr = 0; itr < MAX_ITERATIONS; ++itr) {
		std::cout << "---------- itr #" << (itr + 1) << " ----------" << std::endl;

		// Clear temporary variables.
		double corpus_likelihood = 0;
		memset(&tmp_beta[0], 0, sizeof(double) * tmp_beta.size());
		memset(&sum_beta[0], 0, sizeof(double) * sum_beta.size());

		for (int d = 0; d < NUM_DOCS; ++d) {
			for (int n = 0; n < NUM_VOCABS; ++n) {
				// Update phi.
				double phi_sum = 0;
				for (int k = 0; k < NUM_TOPICS; ++k) {
					const int PIDX = PHI_IDX(d, n, k);
					phi[PIDX] = log_beta[BETA_IDX(k, n)] + digamma(var_gamma[GAMMA_IDX(d, k)]);
					phi_sum = k ? LogSum(phi_sum, phi[PIDX]) : phi[PIDX];
				}

				// Update gamma.
				for (int k = 0; k < NUM_TOPICS; ++k) {
					const int PIDX = PHI_IDX(d, n, k);
					phi[PIDX] = exp(phi[PIDX] - phi_sum);					// Normalize phi.
					const double PHI = corpus[d][n].count * phi[PIDX];

					if (n == 0)
						var_gamma[GAMMA_IDX(d, k)] = ALPHA;
					var_gamma[GAMMA_IDX(d, k)] += PHI;

					tmp_beta[BETA_IDX(k, n)] += PHI;
					sum_beta[k] += PHI;
				}
			}

			// Calculate document log likelihood.
			const double DOC_LIKELIHOOD = CalcDocLikelihood(d, corpus);
			corpus_likelihood += DOC_LIKELIHOOD;
		}

		// Update beta.
		for (int k = 0; k < NUM_TOPICS; ++k) {
			const double LOG_SUM_BETA = log(sum_beta[k]);
			for (int n = 0; n < NUM_VOCABS; ++n) {
				const int BIDX = BETA_IDX(k, n);
				log_beta[BIDX] = log(tmp_beta[BIDX]) - LOG_SUM_BETA;
			}
		}

		// Check convergence.
		std::cout << "Corpus likelihood: " << corpus_likelihood << std::endl;
		const double diff_likelihood = corpus_likelihood - old_likelihood;
		if (diff_likelihood < EPSILON && itr > MIN_ITERATIONS) {
			std::cout << "********** Converged! **********" << std::endl;
			break;
		}

		old_likelihood = corpus_likelihood;
	}

	// Print topics proportions and word probability estimation for each topic.
	PrintGamma(var_gamma);
	PrintBeta(log_beta);
	return 0;
}

