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
	std::unordered_set<std::bitset<N>> words;
	std::string pairs_file_name;
	std::string words_file_name;

public:
	WordsGenerator(std::string words_file_name, std::string pairs_file_name) : words_file_name{ words_file_name }, pairs_file_name{ pairs_file_name } {};
	std::unordered_set<std::bitset<N>> generateWords();
	void generatePairs();

private:
	std::bitset<N> createAlteredWord(std::bitset<N> const& source, int changeBitIndex);
	int hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs);
	
};

template <int N, int M>
std::unordered_set<std::bitset<N>> WordsGenerator<N, M>::generateWords()
{
	Stopwatch stopwatch;
	long long iter = 0;
	auto result = words.emplace(std::bitset<N>());
	std::ofstream wordFile;
	wordFile.open(words_file_name);
	std::queue<std::bitset<N>> generatingWords;
	generatingWords.push(*result.first);
	wordFile << *result.first << std::endl;

	std::cout << "Genrating..." << std::endl;
	stopwatch.Start();
	while (this->words.size() < M)
	{
		std::bitset<N>& generatingWord = generatingWords.front();

		for (size_t i = 0; i < N; i++)
		{
			if (!generatingWord[i])
			{
				iter++;
				auto result = words.emplace(createAlteredWord(generatingWord, i));
				if (result.second)
				{
					generatingWords.push(*result.first);
					wordFile << *result.first << std::endl;
				}
			}
		}
		generatingWords.pop();
	}
	stopwatch.Stop();

	std::cout << "Total iter: " << iter << " words:" << this->words.size() << std::endl;
	return words;
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
int WordsGenerator<N, M>::hammingDistance(std::bitset<N> const& lhs, std::bitset<N> const& rhs)
{
	return (lhs ^ rhs).count();
}

