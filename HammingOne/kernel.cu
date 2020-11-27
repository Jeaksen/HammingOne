
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <math.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <limits>


#include "Stopwatch.h"
#include "WordsGenerator.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


void loadWords();
thrust::device_vector<unsigned int> parseWord(std::string input);
void parseWord(std::string input, thrust::device_vector<unsigned int>& words);
thrust::device_vector<unsigned int> readWords();
void readPairs();

const int word_size = 100;
const int subword_size = 32;
const int subwords_count = (int)ceil( word_size / (double)subword_size);

const std::string words_file_name = "./100-10000/words.csv";
const std::string pairs_file_name = "./100-10000/pairs.csv";

__global__ void searchHammingOne(unsigned int *words, unsigned int *output, unsigned int wordsCount, unsigned int subwords_count, unsigned int ints_per_words_count, unsigned int bits_per_subword, int *foundWordsCount)
{
	int wordIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (wordIndex >= wordsCount)
		return;

	int checkedIndex = wordIndex + 1;
	unsigned int distance;
	int index = wordIndex * ints_per_words_count;

	while (checkedIndex < wordsCount)
	{
		distance = 0;
		for (size_t i = 0; i < subwords_count && distance < 2; i++)
		{
			distance += __popc(words[subwords_count * wordIndex + i] ^ words[subwords_count * checkedIndex + i]);
		}
		if (distance == 1)
		{
			atomicAdd(foundWordsCount, 1);
			int index = wordIndex * ints_per_words_count;
			int offset = checkedIndex / bits_per_subword;
			int value = 1 << bits_per_subword - 1 - checkedIndex % bits_per_subword;
			output[index + offset] += value;
		}
		checkedIndex++;	
	}
}

////#define GENERATE_WORDS
int main()
{	
#ifdef GENERATE_WORDS
	loadWords();
#else
	std::cout << "Reading data...";
	auto words = readWords();
	auto wordsPtr = thrust::raw_pointer_cast(words.begin().base());
	std::cout << " Done!" << std::endl;

	const int words_count = words.size() / subwords_count;
	// adjust to demensions to size of ints - each bits represents one word
	const int ints_per_words_count = ceil(words_count / 32.0);
	const int output_ints_count = words_count * ints_per_words_count;
	const int output_size = output_ints_count * sizeof(int);
	int threads = 512;
	int blocks = (int)ceil(words.size() / (double)threads);
	int *d_count, h_count;
	unsigned int *d_output, *h_output;

	float time;
	cudaEvent_t start, stop; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	cudaMalloc(&d_count, sizeof(int));
	cudaMalloc(&d_output, output_size);
	h_output = new unsigned int[output_ints_count];
	for (size_t i = 0; i < 5; i++)
	{


		cudaMemset(d_count, 0, sizeof(int));
		cudaMemset(d_output, 0, output_size);


		cudaEventRecord(start, 0);
		searchHammingOne <<<blocks, threads>>> (wordsPtr, d_output, words.size() / subwords_count, subwords_count, ints_per_words_count, subword_size, d_count);
		cudaEventRecord(stop, 0);

	
		cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		cudaEventElapsedTime(&time, start, stop);

		std::cout << h_count << std::endl;
		std::cout << "B: " << std::setw(3) << blocks << " T: " << std::setw(4) << threads <<  " time: " << time << " 100-?000" << std::endl;
		h_count = 0;
		int temp;
		for (size_t i = 0; i < output_ints_count; i++)
		{
			temp = __popcnt(h_output[i]);
			h_count += temp;
			//if (i % ints_per_words_count == 0)
			//	std::cout << std::endl;
			//std::cout << temp << " ";
		}
		std::cout << h_count << std::endl;
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_count);
	cudaFree(d_output);
	delete[] h_output;
#endif
	return 0;
}

void loadWords()
{
	auto generator = WordsGenerator(words_file_name, pairs_file_name);
	auto words = generator.generateWords();
	generator.generatePairs();
}

thrust::device_vector<unsigned int> readWords()
{
	std::ifstream wordsStream{ words_file_name };
	//Stopwatch stopwatch;
	std::string word;

	thrust::device_vector<unsigned int> words;

	//stopwatch.Start();
	while (!wordsStream.eof())
	{
		std::getline(wordsStream, word);
		if (!word.empty()) parseWord(word, words);
		//std::cout << "word: " << word << std::endl;
	}

	//stopwatch.Stop();
	wordsStream.close();

	//for (size_t i = 0; i < words.size(); i++)
	//{
	//	if (i % subwords_count == 0)
	//		std::cout << std::endl;
	//	std::cout << words[i] << " ";
	//}

	return words;
}

void readPairs()
{
	std::ifstream pairsStream{ pairs_file_name };
	//Stopwatch stopwatch;
	std::string word;

	//stopwatch.Start();
	while (!pairsStream.eof())
	{
		std::getline(pairsStream, word, ';');
		//parseWord(word);
		//std::cout << "word" << word << std::endl;
		//std::cout << std::endl;
		std::getline(pairsStream, word);
		//parseWord(word);
		//std::cout << word << std::endl;
	}

	//stopwatch.Stop();
	pairsStream.close();
}

void parseWord(std::string input, thrust::device_vector<unsigned int> & words)
{
	// '0' = 48/ 0b0011000
	// '1' = 49/ 0b0011001 
	// Works faste than stoi with base 2!
	const unsigned char zeroAsciiCode = 48;
	
	unsigned int maxValue = pow(2, subword_size - 1);
	unsigned int currentValue = maxValue;
	words.push_back(0);
	auto index = words.size() - 1;

	for (auto letter : input)
	{
		if (!currentValue)
		{
			currentValue = maxValue;
			words.push_back(0);
			//std::cout << word[index] << std::endl;
			index++;
		}
		if (letter - zeroAsciiCode)
			words[index] += currentValue;
		currentValue >>= 1;
	}
}

thrust::device_vector<unsigned int> parseWord(std::string input)
{
	// '0' = 48/ 0b0011000
	// '1' = 49/ 0b0011001 
	// Works faste than stoi with base 2!
	const unsigned char zeroAsciiCode = 48;
	thrust::device_vector<unsigned int> word;

	unsigned int maxValue = pow(2, subword_size - 1);
	unsigned int currentValue = maxValue;
	word.push_back(0);
	auto index = 0;
	for (auto letter : input)
	{
		if (!currentValue)
		{
			currentValue = maxValue;
			word.push_back(0);
			//std::cout << word[index] << std::endl;
			index++;
		}
		if (letter - zeroAsciiCode)
			word[index] += currentValue;
		currentValue >>= 1;
	}

	return word;
}

