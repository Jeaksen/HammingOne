
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

// constants specifying words characteristics

const int word_size = 1000;
const int words_count = 100000;
const int subword_size = 32;
const int subwords_count = (int)ceil(word_size / (double)subword_size);


// Names of files used for saving the words and pairs of words for the generator and the kernel
const std::string words_file_name = std::string("./data/") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("-words.txt");
const std::string pairs_file_name = std::string("./data/") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("-pairs.txt");
const std::string pairs_file_name_kernel = std::string("./data/") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("-kpairs.txt");
const std::string pairs_file_name_kernel_verbose = std::string("./data/") + std::to_string(word_size) + std::string("-") + std::to_string(words_count) + std::string("-kvpairs.csv");

WordsGenerator<word_size, words_count> generator("", "");

// Forward declarations of funcations

std::vector<unsigned> loadWordsForGPU();
std::vector<std::vector<unsigned int>> loadWordsForGPUStride();
thrust::device_vector<unsigned int> copyWordsToGPU(std::vector<unsigned int> generatedWords);
thrust::device_vector<unsigned int> copyWordsToGPUStride(std::vector<std::vector<unsigned int>> generatedWords);
void generatePairsCPU();


/* GPU kernel finding words which are in distance 1
*  words is an array with 32-bit subwords, where each consequtive subword of a word is strided by wordsCount * 4 bits
*  output is an n x ceil(n / 32) matrix, where each row represents a word and each bit of the row is indicating whether word i and j are in distance 1 
*  Each thread represents one word for which it then checks the distance with all words which have higher index that the particular word
*  If the words are in distance 1 the thread sets a corresponding bit in the output array to 1 
*  The distance is calucated as the number of bits set to 1 in the XOR operation of two subwords
*/
__global__ void searchHammingOne(unsigned int* words, unsigned int* output, unsigned int wordsCount, unsigned int subwords_count, unsigned int ints_per_words_count, unsigned int bits_per_subword)
{
	int wordIndex = threadIdx.x + blockIdx.x * blockDim.x;

	if (wordIndex >= wordsCount)
		return;

	unsigned int* word = new unsigned int[subwords_count];

	int checkedIndex = wordIndex + 1;
	unsigned int distance;
	int offset, value, index = wordIndex * ints_per_words_count;

	// First save the word assigned to the thread in local mememory
	// It's going to be used at each itteration of the loop
	for (size_t i = 0; i < subwords_count; i++)
		word[i] = words[wordIndex + i * wordsCount];

	// Then go through all words which have index bigger than the assigned word
	while (checkedIndex < wordsCount)
	{
		distance = 0;
		// Go through all subwords of the checkedIndex word
		for (size_t i = 0; i < subwords_count && distance < 2; i++)
		{
			// add the number of bits set to 1 in the XOR of two subwords
			distance += __popc(word[i] ^ words[checkedIndex + i * words_count]);
		}

		if (distance == 1)
		{
			// the checkedIndex bit from start of the array for the assigned word has to be set
			// it's localted in the checkedIndex / 32 int in the array (the result is automaticaly floored)
			offset = checkedIndex / 32;
			// value represent an UINT with the correspoding bit set to 1
			value = 1 << 31 - checkedIndex % 32;
			// only one bit is set, so the result can be ORed, which is faster than adding
			output[index + offset] |= value;
		}
		checkedIndex++;
	}


	delete[] word;
}


// Uncomment to save pairs generatored by the kernel to a file
//#define VERBOSE_PAIRS
int main()
{
	generator = WordsGenerator<word_size, words_count>(words_file_name, pairs_file_name);

	// Get words from the generator
	std::cout << "Reading data...\n\n";
	auto generatedWords = loadWordsForGPUStride();
	auto words = copyWordsToGPUStride(generatedWords);
	auto wordsPtr = thrust::raw_pointer_cast(words.begin().base());
	//loadPairs();
	std::cout << "Done!" << std::endl;


	//Calculate the size of the output matrix for kernel

	// ints needed to store one row of output - adjusted to size of ints
	const size_t ints_per_words_count = ceil(words_count / 32.0);
	// total number of ints needed to represent the output
	const size_t output_ints_count = words_count * ints_per_words_count;
	// total number of bytes needed to represent the output
	const size_t output_size = output_ints_count * sizeof(int);

	int threads = 256;
	int blocks = (int)ceil(words.size() / (double)threads);

	std::cout << "Words count: " << words_count << std::endl;
	std::cout << "Output size: " << output_size << std::endl;

	unsigned int* d_output, * h_output;
	cudaMalloc(&d_output, output_size);
	h_output = new unsigned int[output_ints_count]();


	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// This stream is used for saving pairs generated by the kernel
	std::ofstream wordPairsFileVerbose;
	wordPairsFileVerbose.open(pairs_file_name_kernel_verbose);


	cudaFuncSetCacheConfig(searchHammingOne, cudaFuncCachePreferL1);
	cudaMemset(d_output, 0, output_size);

	std::cout << "\nCuda start" << std::endl;
	cudaEventRecord(start, 0);
	searchHammingOne <<< blocks, threads >>> (wordsPtr, d_output, words_count, subwords_count, ints_per_words_count, subword_size);
	cudaEventRecord(stop, 0);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));


	cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);

	std::cout << "B: " << std::setw(3) << blocks << " T: " << std::setw(4) << threads << " time: " << time << " ms" << std::endl;

	int parallel_count = 0;

	// This loop counts all 1's set in the output generated in the kernel
	// which correspond to the number of found pairs
	// The loop can also print all pairs to a file
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

	std::cout << "B: " << std::setw(3) << blocks << " T: " << std::setw(4) << threads << " count parallel: " << std::setw(6) << parallel_count << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_output);

	delete[] h_output;
	return 0;
}

// Load words from generator as AoS
std::vector<unsigned> loadWordsForGPU()
{
	return generator.generateWordsForGPU();
}

// Load words from generator as SoA
std::vector<std::vector<unsigned int>> loadWordsForGPUStride()
{
	return generator.generateWordsForGPUStrieded();
}

// Generate pairs using the CPU
// Words must be generated first
void generatePairsCPU()
{
	generator.generatePairs();
}

// Copy AoS data created by the generator to a thrust device vector
thrust::device_vector<unsigned int> copyWordsToGPU(std::vector<unsigned int> generatedWords)
{
	Stopwatch stopwatch;
	thrust::device_vector<unsigned int> words;
	std::cout << "\nCopying to GPU memory" << std::endl;
	stopwatch.Start();

	words.insert(words.begin(), generatedWords.begin(), generatedWords.end());
	stopwatch.Stop();

	return words;
}

// Copy SoA data created by the generator to a thrust device vector
thrust::device_vector<unsigned int> copyWordsToGPUStride(std::vector<std::vector<unsigned int>> generatedWords)
{
	Stopwatch stopwatch;
	thrust::device_vector<unsigned int> words;
	std::cout << "\nCopying to GPU memory" << std::endl;
	stopwatch.Start();

	for (auto & subwords : generatedWords)
		words.insert(words.end(), subwords.begin(), subwords.end());

	stopwatch.Stop();
	std::cout << std::endl;

	return words;
}

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

