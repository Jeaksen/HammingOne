#pragma once
#include <math.h>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <bitset>
#include <queue>
#include <vector>
#include <fstream>
#include "Stopwatch.h"

// This class can generate M unique binary words of length N as a vector of bitsets, or vector of int subwords in the form of AoS and SoA
// It can also generate pairs of words with Hamming distance of 1 with an O(M^2 * N)  sequential algorithm and save the pairs to a file
template <int N, int M>
class WordsGenerator
{
	std::string pairs_file_name;
	std::string words_file_name;

public:
	std::vector<std::bitset<N>> words;

	WordsGenerator(std::string words_file_name, std::string pairs_file_name) : words_file_name{ words_file_name }, pairs_file_name{ pairs_file_name } {};
	// Generate subwords as an AoS
	std::vector<unsigned int> generateWordsForGPU();
	// Generate subwords as an SoA
	std::vector<std::vector<unsigned int>> generateWordsForGPUStrieded();
	// Geberate pairs and save them to a file
	void generatePairs();

private:
	// Given a word create a copy of the word with bit changeBitIndex set to 1
	std::bitset<N> createAlteredWord(std::bitset<N> const& source, int changeBitIndex);
	// Caluclate the Hamming distance between two words
	int hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs);
	// Add the word to the queue and to the AoS vector with subwords
	void saveWord(std::queue<std::bitset<N>>&, std::vector<unsigned int>&, const std::bitset<N>&);
	// Add the word to the queue, to the AoS vector with subwords and to the internal vector of bitsets
	void saveWord(std::queue<std::bitset<N>>&, std::vector<unsigned int>&, std::vector<std::bitset<N>>&, const std::bitset<N>&);
	// Add the word to the queue, to the SoA vector with subwords and to the internal vector of bitsets
	void saveWordStride(std::queue<std::bitset<N>>&, std::vector<std::bitset<N>>&, std::vector<std::vector<unsigned int>>&, const std::bitset<N>&);

};

template <int N, int M>
std::vector<unsigned int> WordsGenerator<N, M>::generateWordsForGPU()
{

	std::queue<std::bitset<N>> generatingWords;
	std::vector<unsigned int> wordsGPU;
	std::unordered_set<std::bitset<N>> wordsSet;

	long long iter = 0;
	Stopwatch stopwatch;
	//std::ofstream wordFile;
	//wordFile.open(words_file_name);

	std::cout << "Genrating..." << std::endl;
	stopwatch.Start();

	// Create a 0 word and add it 
	auto result = wordsSet.emplace(std::bitset<N>());
	saveWord(generatingWords, wordsGPU, words, *result.first);

	// Generate next words until the size is obtained
	// at each iteration take the first element from the queue which stores next generating words
	// and create at possible words from it i.e. switch bits set to 0 and add the new word to the vectors and
	// to the queue as the possible next generating word
	while (wordsSet.size() < M)
	{
		std::bitset<N>& generatingWord = generatingWords.front();

		for (size_t i = 0; i < N; i++)
		{
			if (!generatingWord[i])
			{
				iter++;
				auto result = wordsSet.emplace(createAlteredWord(generatingWord, i));
				if (result.second)
				{
					//wordFile << *result.first << std::endl;
					saveWord(generatingWords, wordsGPU, words, *result.first);
					if (wordsSet.size() == M)
						break;
				}
			}
		}
		generatingWords.pop();
	}
	stopwatch.Stop();
	//wordFile.close();

	std::cout << "Total iter: " << iter << " words:" << wordsSet.size() << std::endl;
	return wordsGPU;
}

template <int N, int M>
std::vector<std::vector<unsigned int>> WordsGenerator<N, M>::generateWordsForGPUStrieded()
{
	std::queue<std::bitset<N>> generatingWords;
	std::vector<std::vector<unsigned int>> wordsGPUStride;
	std::unordered_set<std::bitset<N>> wordsSet;

	// First alocate vectors for subwords - i'th vector stores i'th subwords of each word
	for (size_t i = 0; i < ceil(N / 32.0); i++)
		wordsGPUStride.push_back(std::vector<unsigned int>());

	long long iter = 0;
	Stopwatch stopwatch;
	//std::ofstream wordFile;
	//wordFile.open(words_file_name);

	std::cout << "Genrating..." << std::endl;
	stopwatch.Start();

	// Create a 0 word and add it
	auto result = wordsSet.emplace(std::bitset<N>());
	saveWordStride(generatingWords, words, wordsGPUStride, *result.first);

	// Generate next words until the size is obtained
	// at each iteration take the first element from the queue which stores next generating words
	// and create at possible words from it i.e. switch bits set to 0, add the new word to the vectors and
	// to the queue as the possible next generating word
	while (wordsSet.size() < M)
	{
		std::bitset<N>& generatingWord = generatingWords.front();

		for (size_t i = 0; i < N; i++)
		{
			if (!generatingWord[i])
			{
				iter++;
				auto result = wordsSet.emplace(createAlteredWord(generatingWord, i));
				if (result.second)
				{
					//wordFile << *result.first << std::endl;
					saveWordStride(generatingWords, words, wordsGPUStride, *result.first);
					if (wordsSet.size() == M)
						break;
				}
			}
		}
		generatingWords.pop();
	}
	stopwatch.Stop();
	//wordFile.close();

	std::cout << "Total iter: " << iter << " words:" << wordsSet.size() << std::endl;
	return wordsGPUStride;
}


template <int N, int M>
void WordsGenerator<N, M>::saveWord(std::queue<std::bitset<N>>& generatingWords, std::vector<unsigned int>& wordsGPU, const std::bitset<N>& word)
{
	generatingWords.push(word);
	unsigned int subword = 0;
	// Split the word into 32 bit subwords and save each bit from the bitset as a bit in a UINT 
	for (size_t i = 0; i < N; i++)
	{
		if (i > 0 && i % 32 == 0)
		{
			wordsGPU.emplace_back(subword);
			subword = 0;
		}
		subword |= word[i] << 31 - i % 32;
	}
	wordsGPU.emplace_back(subword);
}

template <int N, int M>
void WordsGenerator<N, M>::saveWord(std::queue<std::bitset<N>>& generatingWords, std::vector<unsigned int>& wordsGPU, std::vector<std::bitset<N>>& words, const std::bitset<N>& word)
{
	saveWord(generatingWords, wordsGPU, word);
	words.emplace_back(word);
}

template <int N, int M>
void WordsGenerator<N, M>::saveWordStride(std::queue<std::bitset<N>>& generatingWords, std::vector<std::bitset<N>>& words, std::vector<std::vector<unsigned int>>& wordsGPU, const std::bitset<N>& word)
{
	generatingWords.push(word);
	words.push_back(word);
	unsigned int subword = 0;
	// Split the word into 32 bit subwords and save each bit from the bitset as a bit in a UINT 
	// each subword is stored in the corresponding vector of subwords
	for (size_t i = 0; i < N; i++)
	{
		if (i > 0 && i % 32 == 0)
		{
			wordsGPU[i / 32 - 1].emplace_back(subword);
			subword = 0;
		}
		subword |= word[i] << 31 - i % 32;
	}
	wordsGPU[wordsGPU.size() - 1].emplace_back(subword);
}

template <int N, int M>
std::bitset<N> WordsGenerator<N, M>::createAlteredWord(std::bitset<N> const& source, int changeBitIndex)
{
	auto result = std::bitset<N>(source);
	result[changeBitIndex] = 1;
	return result;
}

template <int N, int M>
void WordsGenerator<N, M>::generatePairs()
{
	Stopwatch stopwatch;
	std::ofstream wordPairsFile;
	wordPairsFile.open(pairs_file_name);
	int count = 0;
	auto innerIterator = words.begin()++;

	std::cout << "Generating pairs" << std::endl;
	stopwatch.Start();

	// For each word iterate through all words with index bigger than the checked word
	// for each pair of words check their distance and write the pair to the file if the distance between them is 1
	for (auto const& elemDim1 : this->words)
	{
		for (auto it = innerIterator; it != words.end(); it++)
		{
			int dist = hammingDistance(elemDim1, *it);
			if (dist == 1)
			{
				count++;
				wordPairsFile << elemDim1 << ';' << *it << std::endl;
			}
		}
		innerIterator++;
	}

	stopwatch.Stop();
	std::cout << count << " pairs found" << std::endl;
	wordPairsFile.close();
}


template <int N, int M>
int WordsGenerator<N, M>::hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs)
{
	return (lhs ^ rhs).count();
}

