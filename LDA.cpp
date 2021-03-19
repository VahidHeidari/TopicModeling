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

#define PHI_IDX(D, N, K) ((D * NUM_VOCABS * NUM_TOPICS) + (N * NUM_TOPICS) + K)
#define GAMMA_IDX(D, K)  ((D * NUM_TOPICS) + K)
#define BETA_IDX(K, N)   ((K * NUM_VOCABS) + N)



// Constants
constexpr int MAX_ITERATIONS = 100;
constexpr int MIN_ITERATIONS = 2;
constexpr double EPSILON = 5.0;

/// Parameters
constexpr double ALPHA = 0.5;

int NUM_TOPICS = 0;					/// K
int NUM_VOCABS = 0;					/// V
int MAX_WORD_IN_DOC = 0;			/// N
int NUM_DOCS = 0;					/// D
int MIN_TERM = 0;

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
	int min_term = std::numeric_limits<int>::max();
	int max_term = std::numeric_limits<int>::min();
	int max_words_in_doc = std::numeric_limits<int>::min();
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
	}

	// Update some parameters.
	NUM_DOCS = static_cast<int>(corpus.size());
	NUM_VOCABS = max_term - min_term + 1;
	MAX_WORD_IN_DOC = max_words_in_doc;
	MIN_TERM = min_term;
	return true;
}

void DumpBeta(const std::vector<double>& b)
{
	std::ofstream out_file("beta.txt");
	out_file.precision(7);
	out_file << std::fixed;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		out_file << k << " ->   " << exp(b[BETA_IDX(k, 0)]);
		for (int n = 1; n < NUM_VOCABS; ++n)
			out_file << ' ' << exp(b[BETA_IDX(k, n)]);
		if (k + 1 < NUM_TOPICS)
			out_file << std::endl;
	}
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

void DumpGamma(const std::vector<double>& g)
{
	std::ofstream out_file("var_gamma.txt");
	for (int d = 0; d < NUM_DOCS; ++d) {
		int doc_topic = GetDocTopic(d, g);
		out_file << std::setw(3) << d << " ->  " << doc_topic << "  " << g[GAMMA_IDX(d, 0)];
		for (int k = 1; k < NUM_TOPICS; ++k)
			out_file << "   " << g[GAMMA_IDX(d, k)];
		if (d + 1 < NUM_DOCS)
			out_file << std::endl;
	}
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

double CalcDocLogLikelihood(int d, const Corpus& corpus)
{
	double gamma_sum = 0;
	std::vector<double> digamma_gamma(NUM_TOPICS, 0.0);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		const int GIDX = GAMMA_IDX(d, k);
		digamma_gamma[k] = digamma(var_gamma[GIDX]);
		gamma_sum += var_gamma[GIDX];
	}
	double digamma_gamma_sum = digamma(gamma_sum);
	double log_likelihood = - NUM_TOPICS * lgamma(ALPHA) - lgamma(gamma_sum);
	const Doc& doc = corpus[d];
	for (int k = 0; k < NUM_TOPICS; ++k) {
		const double DIG = digamma_gamma[k] - digamma_gamma_sum;
		const int GIDX = GAMMA_IDX(d, k);
		log_likelihood += (ALPHA - 1) * DIG + lgamma(var_gamma[GIDX]) - (var_gamma[GIDX] - 1) * DIG;
		for (int n = 0; n < static_cast<int>(doc.size()); ++n) {
			const int w_n = doc[n].term - MIN_TERM;
			const int PIDX = PHI_IDX(d, w_n, k);
			log_likelihood += phi[PIDX] * (DIG - log(phi[PIDX]) + (doc[n].count * log_beta[BETA_IDX(k, w_n)]));
		}
	}
	return log_likelihood;
}

double LogSum(double log_a, double log_b)
{
	if (log_b < log_a)
		std::swap(log_a, log_b);

	double res = log_b + log(1 + exp(log_a - log_b));
	return res;
}

void CalcAccuracy()
{
	if (NUM_TOPICS > 5)
		return;

	std::vector<std::vector<int>> cluster_count(NUM_TOPICS, std::vector<int>(NUM_TOPICS, 0));

	// Count clusters.
	const int NUM_DOCS_IN_TOPIC = NUM_DOCS / NUM_TOPICS;
	for (int k = 0; k < NUM_TOPICS; ++k)
		for (int d = 0; d < NUM_DOCS_IN_TOPIC; ++d)
			++cluster_count[k][GetDocTopic(k * NUM_DOCS_IN_TOPIC + d, var_gamma)];

	std::vector<int> perm(NUM_TOPICS);
	for (int i = 0; i < NUM_TOPICS; ++i)
		perm[i] = i;

	// Calculate accuracies and find maximum one for report.
	std::vector<double> accs;
	do {
		double sm = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k)
			sm += cluster_count[k][perm[k]];
		const double ACC = sm / static_cast<double>(NUM_DOCS);
		accs.push_back(ACC);
	} while (std::next_permutation(perm.begin(), perm.end()));

	// Print resutl.
	std::cout << " Accuracy: " << *std::max_element(accs.begin(), accs.end()) << "   [ ";
	for (const auto& a : accs)
		std::cout << a << ' ';
	std::cout << ']' << std::endl;
}



int main(int argc, char** argv)
{
	// Check number of command line arguments.
	if (argc <= 2) {
		// Print usage.
		std::cout << "Usage:   " << argv[0] << " CORPUS_PATH   K" << std::endl;
		return 1;
	}

	// Parse command line arguments.
	const char* input_file = argv[1];
	NUM_TOPICS = atoi(argv[2]);

	// Read corpus.
	Corpus corpus;
	if (!ReadCorpus(input_file, corpus)) {
		std::cout << "Could not read `" << input_file << "' file!" << std::endl;
		return 1;
	}

	// Print some statistics.
	std::cout << "Num topics          : " << NUM_TOPICS << std::endl;
	std::cout << "Num vocabularies    : " << NUM_VOCABS << std::endl;
	std::cout << "Corpus size         : " << corpus.size() << std::endl;
	std::cout << "Max num words in doc: " << MAX_WORD_IN_DOC << std::endl;

	// Initialize parameters.
	var_gamma.resize(NUM_DOCS * NUM_TOPICS, ALPHA + (static_cast<double>(MAX_WORD_IN_DOC) / NUM_TOPICS));
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

	double old_likelihood = -std::numeric_limits<double>::max();
	for (int itr = 0; itr < MAX_ITERATIONS; ++itr) {
		std::cout << "---------- itr #" << (itr + 1) << " ----------" << std::endl;

		// Clear temporary variables.
		double corpus_likelihood = 0;
		memset(&tmp_beta[0], 0, sizeof(double) * tmp_beta.size());
		memset(&sum_beta[0], 0, sizeof(double) * sum_beta.size());

		// Iterate over documents and update variational parameters.
		for (int d = 0; d < NUM_DOCS; ++d) {
			const Doc& doc = corpus[d];
			for (int n = 0; n < static_cast<int>(doc.size()); ++n) {
				const int w_n = doc[n].term - MIN_TERM;						// Transform term number to 0-based index.

				// Update phi.
				double phi_sum = 0;
				for (int k = 0; k < NUM_TOPICS; ++k) {
					const int PIDX = PHI_IDX(d, w_n, k);
					phi[PIDX] = log_beta[BETA_IDX(k, w_n)] + digamma(var_gamma[GAMMA_IDX(d, k)]);
					phi_sum = k ? LogSum(phi_sum, phi[PIDX]) : phi[PIDX];
				}

				// Update gamma.
				for (int k = 0; k < NUM_TOPICS; ++k) {
					const int PIDX = PHI_IDX(d, w_n, k);
					phi[PIDX] = exp(phi[PIDX] - phi_sum);					// Normalize phi.
					const double PHI = doc[n].count * phi[PIDX];

					if (n == 0)
						var_gamma[GAMMA_IDX(d, k)] = ALPHA;
					var_gamma[GAMMA_IDX(d, k)] += PHI;

					tmp_beta[BETA_IDX(k, w_n)] += PHI;
					sum_beta[k] += PHI;
				}
			}

			// Calculate document log likelihood.
			const double DOC_LIKELIHOOD = CalcDocLogLikelihood(d, corpus);
			corpus_likelihood += DOC_LIKELIHOOD;
		}

		// Update model parameter beta.
		for (int k = 0; k < NUM_TOPICS; ++k) {
			const double LOG_SUM_BETA = log(sum_beta[k]);
			for (int n = 0; n < NUM_VOCABS; ++n) {
				const int BIDX = BETA_IDX(k, n);
				log_beta[BIDX] = log(tmp_beta[BIDX]) - LOG_SUM_BETA;
				log_beta[BIDX] = (std::isinf(-log_beta[BIDX]) || log_beta[BIDX] != log_beta[BIDX]) ? -100 : log_beta[BIDX];
			}
		}

		// Check convergence.
		std::cout << "Corpus likelihood: " << corpus_likelihood << std::endl;
		const double diff_likelihood = corpus_likelihood - old_likelihood;
		std::cout << "diff likelihood:   " << diff_likelihood << std::endl;
		if (diff_likelihood < EPSILON && itr > MIN_ITERATIONS) {
			std::cout << "********** Converged! **********" << std::endl;
			std::cout << "diff likelihood: " << diff_likelihood << std::endl;
			std::cout << "********************************" << std::endl;
			break;
		}

		old_likelihood = corpus_likelihood;
	}

	// Write topics proportions and word probability estimation for each topic.
	DumpGamma(var_gamma);
	DumpBeta(log_beta);
	CalcAccuracy();
	return 0;
}

