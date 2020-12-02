
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

const int word_size = 100;
const int words_count = 10000;
const int subword_size = 32;
const int subwords_count = (int)ceil(word_size / (double)subword_size);

const std::string words_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("/words.txt");
const std::string pairs_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("/pairs.txt");
const std::string pairs_file_name_kernel = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("/kpairs.txt");
const std::string pairs_file_name_kernel_verbose = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("/kvpairs.csv");

WordsGenerator<word_size, words_count> generator("", "");

std::vector<unsigned> loadWordsForGPU();
std::vector<std::vector<unsigned int>> loadWordsForGPUStride();
thrust::device_vector<unsigned int> copyWordsToGPU(std::vector<unsigned int> generatedWords);
thrust::device_vector<unsigned int> copyWordsToGPUStride(std::vector<std::vector<unsigned int>> generatedWords);
void loadPairs(unsigned int* output, int output_size, int stride);
//void readPairs();
//std::unordered_set<std::bitset<word_size>> loadWords();
//thrust::device_vector<unsigned int> parseWord(std::string input);
//void parseWord(std::string input, thrust::device_vector<unsigned int>& words);
//thrust::device_vector<unsigned int> copyWordsToGPU();
//thrust::device_vector<unsigned int> copyWordsToGPU(std::unordered_set<std::bitset<word_size>> generatedWords);

__global__ void searchHammingOne(unsigned int* words, unsigned int* output, unsigned int wordsCount, unsigned int subwords_count, unsigned int ints_per_words_count, unsigned int bits_per_subword)
{
	int wordIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (wordIndex >= wordsCount)
		return;

	unsigned int* word = new unsigned int[subwords_count];

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
			offset = checkedIndex / bits_per_subword;
			value = 1 << bits_per_subword - 1 - checkedIndex % bits_per_subword;
			output[index + offset] |= value;
		}
		checkedIndex++;
	}

	delete[] word;
}

//#define VERBOSE_PAIRS
int main()
{
	generator = WordsGenerator<word_size, words_count>(words_file_name, pairs_file_name);
	std::cout << "Reading data..." << std::endl;
	//auto generatedWords = loadWordsForGPU();
	//auto words = copyWordsToGPU(generatedWords);
	auto generatedWords = loadWordsForGPUStride();
	auto words = copyWordsToGPUStride(generatedWords);
	auto wordsPtr = thrust::raw_pointer_cast(words.begin().base());
	std::cout << "Done!" << std::endl;


	// adjust the demensions to size of ints - each bits represents one word
	const int ints_per_words_count = ceil(words_count / 32.0);
	const int output_ints_count = words_count * ints_per_words_count;
	const int output_size = output_ints_count * sizeof(int);
	int threads = 512;
	int blocks = (int)ceil(words.size() / (double)threads);

	std::cout << "Words count: " << words_count << std::endl;
	std::cout << "Output size: " << output_size << std::endl;

	unsigned int* d_output, * h_output;
	cudaMalloc(&d_output, output_size);
	h_output = new unsigned int[output_ints_count]();

	//loadPairs(h_output, output_size, ints_per_words_count);

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//std::ofstream wordPairsFile;
	//wordPairsFile.open(pairs_file_name_kernel);
	std::ofstream wordPairsFileVerbose;
	wordPairsFileVerbose.open(pairs_file_name_kernel_verbose);

	//for (size_t i = 0; i < 3; i++)
	//{
		//cudaFuncSetCacheConfig(searchHammingOne, cudaFuncCachePreferL1);
	cudaMemset(d_output, 0, output_size);


	cudaEventRecord(start, 0);
	searchHammingOne << <blocks, threads >> > (wordsPtr, d_output, words_count, subwords_count, ints_per_words_count, subword_size);
	cudaEventRecord(stop, 0);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);

	//wordPairsFile.write((char*)h_output, output_size);

	int parallel_count = 0;

	for (size_t i = 0; i < words_count; i++)
	{
		for (size_t j = 0; j < ints_per_words_count; j++)
		{
			parallel_count += __popcnt(h_output[i * ints_per_words_count + j]);
#ifdef VERBOSE_PAIRS
			for (int k = 0; k < subword_size; k++)
			{
				if (h_output[i * ints_per_words_count + j] & 1 << 31 - k)
					wordPairsFileVerbose << generator.words[i] << ';' << generator.words[j * subword_size + k] << std::endl;
			}
#endif
		}
	}

	std::cout << "B: " << std::setw(3) << blocks << " T: " << std::setw(4) << threads << " count parallel: " << std::setw(6) << parallel_count << " time: " << time << std::endl;
	//}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_output);
	//wordPairsFile.close();

	delete[] h_output;
	return 0;
}


std::vector<unsigned> loadWordsForGPU()
{
	return generator.generateWordsForGPU();
}

std::vector<std::vector<unsigned int>> loadWordsForGPUStride()
{
	return generator.generateWordsForGPUStrieded();
}


void loadPairs(unsigned int* output, int output_size, int stride)
{
	return generator.generatePairsForGPU(output, output_size, stride);
}

thrust::device_vector<unsigned int> copyWordsToGPU(std::vector<unsigned int> generatedWords)
{
	Stopwatch stopwatch;
	thrust::device_vector<unsigned int> words;
	std::cout << "Copying to GPU memory" << std::endl;
	stopwatch.Start();

	words.insert(words.begin(), generatedWords.begin(), generatedWords.end());
	stopwatch.Stop();

	return words;
}

thrust::device_vector<unsigned int> copyWordsToGPUStride(std::vector<std::vector<unsigned int>> generatedWords)
{
	Stopwatch stopwatch;
	thrust::device_vector<unsigned int> words;
	std::cout << "Copying to GPU memory" << std::endl;
	stopwatch.Start();

	for (auto & subwords : generatedWords)
	{
		//if (words.size() == 0)
		//	words.insert(words.begin(), subwords.begin(), subwords.end());
		//else 
			words.insert(words.end(), subwords.begin(), subwords.end());

	}

	stopwatch.Stop();

	return words;
}


//void readPairs()
//{
//	std::ifstream pairsStream{ pairs_file_name };
//	//Stopwatch stopwatch;
//	std::string word;
//
//	//stopwatch.Start();
//	while (!pairsStream.eof())
//	{
//		std::getline(pairsStream, word, ';');
//		//parseWord(word);
//		//std::cout << "word" << word << std::endl;
//		//std::cout << std::endl;
//		std::getline(pairsStream, word);
//		//parseWord(word);
//		//std::cout << word << std::endl;
//	}
//
//	//stopwatch.Stop();
//	pairsStream.close();
//}

//thrust::device_vector<unsigned int> parseWord(std::string input)
//{
//	// '0' = 48/ 0b0011000
//	// '1' = 49/ 0b0011001 
//	// Works faste than stoi with base 2!
//	const unsigned char zeroAsciiCode = 48;
//	thrust::device_vector<unsigned int> word;
//
//	unsigned int maxValue = pow(2, subword_size - 1);
//	unsigned int currentValue = maxValue;
//	word.push_back(0);
//	auto index = 0;
//	for (auto letter : input)
//	{
//		if (!currentValue)
//		{
//			currentValue = maxValue;
//			word.push_back(0);
//			//std::cout << word[index] << std::endl;
//			index++;
//		}
//		if (letter - zeroAsciiCode)
//			word[index] += currentValue;
//		currentValue >>= 1;
//	}
//
//	return word;
//}

//std::unordered_set<std::bitset<word_size>> loadWords()
//{
//	auto generator = WordsGenerator<word_size, words_count>(words_file_name, pairs_file_name);
//	return generator.generateWords();
//	//generator.generatePairs();
//}

//thrust::device_vector<unsigned int> copyWordsToGPU(std::unordered_set<std::bitset<word_size>> generatedWords)
//{
//
//	Stopwatch stopwatch;
//
//	thrust::device_vector<unsigned int> words;
//
//	stopwatch.Start();
//	unsigned int subword;
//	for (auto word : generatedWords)
//	{
//		subword = 0;
//		for (size_t i = 0; i < word_size; i++)
//		{
//			if (i > 0 && i % subword_size == 0)
//			{
//				words.push_back(subword);
//				subword = 0;
//			}
//			subword |= word[i] << subword_size - 1 - i % subword_size;
//		}
//		words.push_back(subword);
//	}
//	stopwatch.Stop();
//
//	return words;
//}

//thrust::device_vector<unsigned int> copyWordsToGPU()
//{
//	std::ifstream wordsStream{ words_file_name };
//	Stopwatch stopwatch;
//	std::string word;
//	thrust::device_vector<unsigned int> words;
//
//	stopwatch.Start();
//	while (!wordsStream.eof())
//	{
//		std::getline(wordsStream, word);
//		if (!word.empty()) parseWord(word, words);
//		//std::cout << "word: " << word << std::endl;
//	}
//
//	stopwatch.Stop();
//	wordsStream.close();
//	return words;
//}

//void parseWord(std::string input, thrust::device_vector<unsigned int>& words)
//{
//	// '0' = 48/ 0b0011000
//	// '1' = 49/ 0b0011001 
//	// Works faste than stoi with base 2!
//	const unsigned char zeroAsciiCode = 48;
//
//	unsigned int maxValue = pow(2, subword_size - 1);
//	unsigned int currentValue = maxValue;
//	words.push_back(0);
//	auto index = words.size() - 1;
//
//	for (auto letter : input)
//	{
//		if (!currentValue)
//		{
//			currentValue = maxValue;
//			words.push_back(0);
//			//std::cout << word[index] << std::endl;
//			index++;
//		}
//		if (letter - zeroAsciiCode)
//			words[index] += currentValue;
//		currentValue >>= 1;
//	}
//}

