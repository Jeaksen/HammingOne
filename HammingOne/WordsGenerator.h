#pragma once
#include <math.h>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <bitset>

const int my_word_size = 100;
const int minimal_words_count = 10000;
#define words_set std::unordered_set<std::bitset<my_word_size>>
//#define words_pairs_vector std::vector < std::pair < std::bitset<my_word_size> const, std::bitset<my_word_size> const>>

class WordsGenerator 
{
	words_set words;
	std::string pairs_file_name;
	std::string words_file_name;

public:
	WordsGenerator(std::string words_file_name, std::string pairs_file_name) : words_file_name{ words_file_name }, pairs_file_name{ pairs_file_name } {};
	words_set generateWords();
	void generatePairs();

private:
	std::bitset<my_word_size> createAlteredWord(std::bitset<my_word_size> const& source, int changeBitIndex);
	int hammingDistance(std::bitset<my_word_size> const& lhs, std::bitset<my_word_size> const& rhs);
	
};

