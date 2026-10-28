// Byte Pair Encoding (BPE) tokenizer: translates strings <-> token ids
// The vocabulary is stored in conf_and_weights_fp16.c (VOCAB_DATA, VOCAB_SCORES, SORTED_VOCAB)
#include "conf_and_weights_fp16.h"
#include <string.h>
#include <ctype.h>
#include "stdlib.h"
#include "stdio.h"

#ifndef _TokenIndex_
#define _TokenIndex_
// entry of the vocabulary sorted alphabetically, used to look up the id of a string
typedef struct {
    char *str;
    int id;
} TokenIndex;
#endif


typedef struct {
    char** vocab;               // string of each token id (vocab_size,)
    fp16* vocab_scores;         // merge score of each token id (vocab_size,)
    TokenIndex *sorted_vocab;   // vocabulary sorted by string, for binary search (vocab_size,)
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // all 256 single-byte strings, each followed by '\0'
} Tokenizer;



// set up the tokenizer from the vocabulary stored in L2
void build_tokenizer(Tokenizer* t, int vocab_size);

// return the string of token, given the previous token
char* decode(Tokenizer* t, int prev_token, int token);

// print piece, skipping raw bytes that are not printable
void safe_printf(char *piece);

// compare two TokenIndex entries by their string (for bsearch)
int compare_tokens(const void *a, const void *b);

// binary search in a sorted array (own implementation, the libc one is not available on the target)
void *bsearch(const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

// return the id of str in the vocabulary, or -1 if not found
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);

// encode text into tokens[], prepending BOS if bos != 0 and appending EOS if eos != 0
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);