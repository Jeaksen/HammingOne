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

template <int N, int M>
class WordsGenerator
{
	std::string pairs_file_name;
	std::string words_file_name;

public:
	std::vector<std::bitset<N>> words;

	WordsGenerator(std::string words_file_name, std::string pairs_file_name) : words_file_name{ words_file_name }, pairs_file_name{ pairs_file_name } {};
	//std::unordered_set<std::bitset<N>> generateWords();
	std::vector<unsigned int> generateWordsForGPU();
	void WordsGenerator<N, M>::generatePairsForGPU(unsigned int* output, int output_size, int stride);
	void generatePairs();
private:
	std::bitset<N> createAlteredWord(std::bitset<N> const& source, int changeBitIndex);
	int hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs);
	void saveWord(std::queue<std::bitset<N>>&, std::vector<unsigned int>&, const std::bitset<N>&);
	void saveWord(std::queue<std::bitset<N>>&, std::vector<unsigned int>&, std::vector<std::bitset<N>>&, const std::bitset<N>&);

};

//template <int N, int M>
//std::unordered_set<std::bitset<N>> WordsGenerator<N, M>::generateWords()
//{
//	Stopwatch stopwatch;
//	long long iter = 0;
//	auto result = words.emplace(std::bitset<N>());
//	std::ofstream wordFile;
//	wordFile.open(words_file_name);
//	std::queue<std::bitset<N>> generatingWords;
//	generatingWords.push(*result.first);
//	wordFile << *result.first << std::endl;
//
//	std::cout << "Genrating..." << std::endl;
//	stopwatch.Start();
//	while (this->words.size() < M)
//	{
//		std::bitset<N>& generatingWord = generatingWords.front();
//
//		for (size_t i = 0; i < N; i++)
//		{
//			if (!generatingWord[i])
//			{
//				iter++;
//				auto result = words.emplace(createAlteredWord(generatingWord, i));
//				if (result.second)
//				{
//					generatingWords.push(*result.first);
//					wordFile << *result.first << std::endl;
//				}
//			}
//		}
//		generatingWords.pop();
//	}
//	stopwatch.Stop();
//	wordFile.close();
//
//	std::cout << "Total iter: " << iter << " words:" << this->words.size() << std::endl;
//	return words;
//}

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

	auto result = wordsSet.emplace(std::bitset<N>());
	saveWord(generatingWords, wordsGPU, words, *result.first);

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
void WordsGenerator<N, M>::saveWord(std::queue<std::bitset<N>>& generatingWords, std::vector<unsigned int>& wordsGPU, const std::bitset<N>& word)
{
	generatingWords.push(word);
	unsigned int subword = 0;
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
void WordsGenerator<N, M>::generatePairsForGPU(unsigned int* output, int output_size, int stride)
{
	Stopwatch stopwatch;
	std::ofstream wordPairsFile;
	wordPairsFile.open(pairs_file_name);
	auto innerIterator = words.begin()++;

	int wordIdx = 0, count = 0;

	std::cout << "Generating pairs" << std::endl;
	stopwatch.Start();

	for (auto const& elemDim1 : this->words)
	{
		for (auto it = innerIterator; it != words.end(); it++)
		{

			int dist = hammingDistance(elemDim1, *it);
			if (dist == 1)
			{
				//int index = it - words.begin();
				count++;
				wordPairsFile << elemDim1 << ';' << *it << std::endl;
				//output[wordIdx * stride + index / 32] += 1 << 31 - index % 32;
			}

		}
		innerIterator++;
		wordIdx++;
		//wordPairsFile << std::endl;
	}
	//wordPairsFile.write((char*)output, output_size);
	stopwatch.Stop();
	std::cout << count << " pairs found" << std::endl;

	wordPairsFile.close();
}

template <int N, int M>
int WordsGenerator<N, M>::hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs)
{
	return (lhs ^ rhs).count();
}

