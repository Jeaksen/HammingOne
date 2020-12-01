
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

#define BLOCK_SIZE 32
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
	unsigned int width;
	unsigned int height;
	unsigned int stride;
	unsigned int* elements;
} Matrix;

const int word_size = 100;
const int min_words_count = 10000;
const int subword_size = 32;
const int subwords_count = (int)ceil( word_size / (double)subword_size);

const std::string words_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(min_words_count) + std::string("/words.csv");
const std::string pairs_file_name = std::string("./") + std::to_string(word_size) + std::string("-") + std::to_string(min_words_count) + std::string("/pairs.csv");

__global__ void MatMulKernel(const Matrix, Matrix);
std::unordered_set<std::bitset<word_size>> loadWords();
std::vector<unsigned> loadWordsForGPU();
thrust::device_vector<unsigned int> parseWord(std::string input);
void parseWord(std::string input, thrust::device_vector<unsigned int>& words);
thrust::device_vector<unsigned int> readWords();
thrust::device_vector<unsigned int> readWords(std::unordered_set<std::bitset<word_size>> generatedWords);
thrust::device_vector<unsigned int> readWords(std::vector<unsigned int> generatedWords);
void readPairs();


__device__ int GetElement(const Matrix A, unsigned int row, unsigned int col)
{
	return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, unsigned int row, unsigned int col, unsigned int value)
{
	// change!!!
	//if (value == 1)
		//A.elements[row * A.stride] |= 1 << 32 - 1 - col % 32;
	//A.elements[row * A.stride + col / 32] |= 1 << 32 - 1 - col % 32;
	if (value == 1)
		atomicAdd(A.elements + row * A.stride + col / 32, 1 << 32 - 1 - col % 32);
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, unsigned int row, unsigned int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

__device__ Matrix GetCSubMatrix(Matrix A, unsigned int row, unsigned int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * row + col];
	return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix C)
{
	// Block row and column
	unsigned int blockRow = blockIdx.y;
	unsigned int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetCSubMatrix(C, blockRow, blockCol);

	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	unsigned int Cvalue = 0;

	// Thread row and column within Csub
	unsigned int row = threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int globalRow = row + blockRow * blockDim.y;
	unsigned int globalCol = col + blockCol * blockDim.x;

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results

	unsigned int limit = ceil(A.width / (double)BLOCK_SIZE);

	for (int m = 0; m < limit; m++)
	{
		__syncthreads();
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(A, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		if (m * BLOCK_SIZE + threadIdx.x < A.width && row < A.height)
			As[row][col] = GetElement(Asub, row, col);
		if (m * BLOCK_SIZE + threadIdx.y < A.height && col < A.width)
			Bs[row][col] = GetElement(Bsub, row, col);


		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();

		if (globalRow >= A.height || globalCol >= A.width)
			continue;

		// Multiply Asub and Bsub together
		for (unsigned int e = 0; e < BLOCK_SIZE; ++e)
		{
			if (m * BLOCK_SIZE + e >= A.width || m * BLOCK_SIZE + e >= A.height)
				break;

			Cvalue += __popc(As[row][e] ^ Bs[e][col]);
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration

	}

	if (globalRow < A.height && globalCol < A.width)
		SetElement(Csub, row, col, Cvalue);
}


//width -> subwords count
//height -> words count
//d_elements -> device pointer to subwords
float MatMul(int width, int height, unsigned int* d_elements, unsigned int* output)
{
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = width; d_A.height = height;
	d_A.elements = d_elements;

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = height; d_C.height = height;
	d_C.elements = output;
	d_C.stride = (int)ceil(height / 32.0);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	unsigned int gridSize = (int)ceil(height / (double)dimBlock.x);
	dim3 dimGrid(gridSize, gridSize);

	cudaEventRecord(start, 0);
	MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_C);
	cudaEventRecord(stop, 0);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}


int main()
{	
	//----------------------------------------------------------------------------------------
	// Read words
	//----------------------------------------------------------------------------------------
	std::cout << "Reading data...\n\n" ;
	auto generatedWords = loadWordsForGPU();
	auto words = readWords(generatedWords);
	auto wordsPtr = thrust::raw_pointer_cast(words.begin().base());
	std::cout << "Done!\n\n";

	//----------------------------------------------------------------------------------------
	// Create output
	//----------------------------------------------------------------------------------------
	const int words_count = words.size() / subwords_count;
	// adjust the demensions to size of ints - each bits represents one word
	const int ints_per_words_count = ceil(words_count / 32.0);
	const int output_ints_count = words_count * ints_per_words_count;
	const int output_size = output_ints_count * sizeof(int);

	std::cout << "Words count: " << words_count << std::endl;
	std::cout << "Output size: " << output_size << std::endl << std::endl;

	float time;
	unsigned int* d_output, * h_output;
	cudaMalloc(&d_output, output_size);
	h_output = new unsigned int[output_ints_count];

	cudaMemset(d_output, 0, output_size);
	cudaDeviceSynchronize();

	//----------------------------------------------------------------------------------------
	// Run Kernel
	//----------------------------------------------------------------------------------------

	time = MatMul(subwords_count, words_count, wordsPtr, d_output);

	//----------------------------------------------------------------------------------------
	// Copy after kernel exec
	//----------------------------------------------------------------------------------------
	cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//----------------------------------------------------------------------------------------
	// Count pairs
	//----------------------------------------------------------------------------------------
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

	std::cout << "Count : " << std::setw(6) << parallel_count <<  " time: " << time << std::endl;


	//----------------------------------------------------------------------------------------
	// Cleanup
	//----------------------------------------------------------------------------------------

	cudaFree(d_output);
	delete[] h_output;
	return 0;
}

__global__ void searchHammingOne(unsigned int* words, unsigned int* output, unsigned int wordsCount, unsigned int subwords_count, unsigned int ints_per_words_count, unsigned int bits_per_subword, int* foundWordsCount)
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
			//atomicAdd(foundWordsCount, 1);
			offset = checkedIndex / bits_per_subword;
			value = 1 << bits_per_subword - 1 - checkedIndex % bits_per_subword;
			output[index + offset] |= value;
		}
		checkedIndex++;
	}

	delete[] word;
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

