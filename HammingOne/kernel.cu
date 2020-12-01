
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

const int word_size = 1000;
const int min_words_count = 100000;
const int subword_size = 32;
const int subwords_count = (int)ceil( word_size / (double)subword_size);

const std::string words_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(min_words_count) + std::string("/words.csv");
const std::string pairs_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(min_words_count) + std::string("/pairs.csv");

std::unordered_set<std::bitset<word_size>> loadWords();
std::vector<unsigned> loadWordsForGPU();
thrust::device_vector<unsigned int> parseWord(std::string input);
void parseWord(std::string input, thrust::device_vector<unsigned int>& words);
thrust::device_vector<unsigned int> readWords();
thrust::device_vector<unsigned int> readWords(std::unordered_set<std::bitset<word_size>> generatedWords);
thrust::device_vector<unsigned int> readWords(std::vector<unsigned int> generatedWords);
void readPairs();

__global__ void searchHammingOne(unsigned int *words, unsigned int *output, unsigned int wordsCount, unsigned int subwords_count, unsigned int ints_per_words_count, unsigned int bits_per_subword, int *foundWordsCount)
{
	int wordIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (wordIndex >= wordsCount)
		return;

	unsigned int *word = new unsigned int [subwords_count];

	for (size_t i = 0; i < subwords_count; i++)
	{
		word[i] = words[wordIndex * subwords_count + i];
	}

	int checkedIndex = wordIndex + 1;
	unsigned int distance;
	int offset, value, index = wordIndex * ints_per_words_count;

	while (checkedIndex < wordsCount)
	{
		distance = 0;
		for (size_t i = 0; i < subwords_count && distance < 2; i++)
		{
			distance += __popc(word[i] ^ words[subwords_count * checkedIndex + i]);
		}
		if (!(distance >> 1))
		{
			//atomicAdd(foundWordsCount, 1);
			offset = checkedIndex / bits_per_subword;
			value = 1 << bits_per_subword - 1 - checkedIndex % bits_per_subword;
			output[index + offset] |= value;
		}
		checkedIndex++;
	}

	delete[] word;
}



int main()
{	
	std::cout << "Reading data..." << std::endl;
	auto generatedWords = loadWordsForGPU();
	auto words = readWords(generatedWords);
	auto wordsPtr = thrust::raw_pointer_cast(words.begin().base());
	std::cout << "Done!" << std::endl;

	const int words_count = words.size() / subwords_count;
	// adjust the demensions to size of ints - each bits represents one word
	const int ints_per_words_count = ceil(words_count / 32.0);
	const int output_ints_count = words_count * ints_per_words_count;
	const int output_size = output_ints_count * sizeof(int);
	int threads = 512;
	int blocks = (int)ceil(words.size() / (double)threads);
	int *d_count, h_count;
	unsigned int *d_output, *h_output;

	std::cout << "Words count: " << words_count << std::endl;
	std::cout << "Output size: " << output_size << std::endl;

	float time;
	cudaEvent_t start, stop; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	cudaMalloc(&d_count, sizeof(int));
	cudaMalloc(&d_output, output_size);
	h_output = new unsigned int[output_ints_count];
	for (size_t i = 0; i < 10; i++)
	{

		cudaMemset(d_count, 0, sizeof(int));
		cudaMemset(d_output, 0, output_size);


		cudaEventRecord(start, 0);
		searchHammingOne <<<blocks, threads>>> (wordsPtr, d_output, words.size() / subwords_count, subwords_count, ints_per_words_count, subword_size, d_count);
		cudaEventRecord(stop, 0);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	
		cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		cudaEventElapsedTime(&time, start, stop);

		int parallel_count= 0;
		int temp;
		for (size_t i = 0; i < output_ints_count; i++)
		{
			temp = __popcnt(h_output[i]);
			parallel_count += temp;
			//if (i % ints_per_words_count == 0)
			//	std::cout << std::endl;
			//std::cout << temp << " ";
		}

		std::cout << "B: " << std::setw(3) << blocks << " T: " << std::setw(4) << threads << " count atomic: " << std::setw(6) << h_count << " count parallel: " << std::setw(6) << parallel_count <<  " time: " << time << " 100-20000" << std::endl;
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_count);
	cudaFree(d_output);
	delete[] h_output;
	return 0;
}


std::vector<unsigned> loadWordsForGPU()
{
	auto generator = WordsGenerator<word_size, min_words_count>(words_file_name, pairs_file_name);
	return generator.generateWordsForGPU();
}

thrust::device_vector<unsigned int> readWords(std::vector<unsigned int> generatedWords)
{
	Stopwatch stopwatch;
	thrust::device_vector<unsigned int> words;
	std::cout << "Copying to GPU memory" << std::endl;
	stopwatch.Start();
	words.insert(words.begin(), generatedWords.begin(), generatedWords.end());
	stopwatch.Stop();

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

std::unordered_set<std::bitset<word_size>> loadWords()
{
	auto generator = WordsGenerator<word_size, min_words_count>(words_file_name, pairs_file_name);
	return generator.generateWords();
	//generator.generatePairs();
}

thrust::device_vector<unsigned int> readWords(std::unordered_set<std::bitset<word_size>> generatedWords)
{

	Stopwatch stopwatch;

	thrust::device_vector<unsigned int> words;

	stopwatch.Start();
	unsigned int subword;
	for (auto word : generatedWords)
	{
		subword = 0;
		for (size_t i = 0; i < word_size; i++)
		{
			if (i > 0 && i % subword_size == 0)
			{
				words.push_back(subword);
				subword = 0;
			}
			subword |= word[i] << subword_size - 1 - i % subword_size;
		}
		words.push_back(subword);
	}
	stopwatch.Stop();

	return words;
}

thrust::device_vector<unsigned int> readWords()
{
	std::ifstream wordsStream{ words_file_name };
	Stopwatch stopwatch;
	std::string word;
	thrust::device_vector<unsigned int> words;

	stopwatch.Start();
	while (!wordsStream.eof())
	{
		std::getline(wordsStream, word);
		if (!word.empty()) parseWord(word, words);
		//std::cout << "word: " << word << std::endl;
	}

	stopwatch.Stop();
	wordsStream.close();
	return words;
}

void parseWord(std::string input, thrust::device_vector<unsigned int>& words)
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

