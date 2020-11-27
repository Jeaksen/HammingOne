#include "WordsGenerator.h"
#include <queue>
#include <vector>
#include <fstream>
#include "Stopwatch.h"

words_set WordsGenerator::generateWords()
{
	Stopwatch stopwatch;
	long long iter = 0;
	auto result = words.emplace(std::bitset<my_word_size>());
	std::ofstream wordFile;
	wordFile.open(words_file_name);
	std::queue<std::bitset<my_word_size>> generatingWords;
	generatingWords.push(*result.first);
	wordFile << *result.first << std::endl;

	std::cout << "Genrating..." << std::endl;
	stopwatch.Start();
	while (this->words.size() < minimal_words_count)
	{
		std::bitset<my_word_size>& generatingWord = generatingWords.front();
		
		for (size_t i = 0; i < my_word_size; i++)
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

std::bitset<my_word_size> WordsGenerator::createAlteredWord(std::bitset<my_word_size> const& source, int changeBitIndex)
{
	auto result = std::bitset<my_word_size>(source);
	result[changeBitIndex] = 1;
	return result;
}

void WordsGenerator::generatePairs()
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

int WordsGenerator::hammingDistance(std::bitset<my_word_size> const& lhs, std::bitset<my_word_size> const& rhs)
{
	return (lhs ^ rhs).count();
}