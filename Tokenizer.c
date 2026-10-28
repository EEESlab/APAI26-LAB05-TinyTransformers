
#include "Tokenizer.h"


void build_tokenizer(Tokenizer* t, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = VOCAB;
    t->vocab_scores = VOCAB_SCORES;
    t->sorted_vocab = SORTED_VOCAB;

    // single-byte strings returned by decode() for the raw byte tokens <0x00>..<0xFF>
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    t->max_token_length = MAX_TOKEN_LENGTH;
    // VOCAB_DATA holds all token strings one after the other, each terminated by '\0':
    // point vocab[i] to the start of the i-th string
    int j=0;
    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = &VOCAB_DATA[j];
        while(VOCAB_DATA[j] != '\0' && i < vocab_size-1)
            j++;
        j++;
    }
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse the two hex digits by hand (instead of sscanf) and return the actual byte
    unsigned char byte_val;
    if(piece[0]=='<' && piece[1] == '0' && piece[2]=='x' && piece[5]=='>'){
        int cifra1, cifra2;
        if('0' <= piece[3] && piece[3]<= '9')
            cifra1 = piece[3] - '0';
        else
            cifra1 = piece[3] - 'A' + 10; 
        if('0' <= piece[4] && piece[4] <= '9')
            cifra2 = piece[4] - '0';
        else
            cifra2 = piece[4] - 'A' + 10;
        byte_val = cifra1*16 + cifra2;

        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}


int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
	const char *base = (const char *) base0;
	int lim, cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return (NULL);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
        exit(1);
    // temporary buffer for the merge candidates of two consecutive tokens
    // it needs max_token_length*2 + 1 bytes: BUFF1 (L1) is large enough and is not in use during encoding
    char* str_buffer = (char* ) BUFF1;
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // like sentencepiece (add_dummy_prefix), prepend a space token to a non-empty text
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // UTF-8 encodes a code point in 1 to 4 bytes:
    // First code point   Last code point   Byte 1     Byte 2     Byte 3     Byte 4
    // U+0000             U+007F            0xxxxxxx
    // U+0080             U+07FF            110xxxxx   10xxxxxx
    // U+0800             U+FFFF            1110xxxx   10xxxxxx   10xxxxxx
    // U+10000            U+10FFFF          11110xxx   10xxxxxx   10xxxxxx   10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // continuation bytes start with the bits 10 (checked with the mask 0xC0 = 11000000):
        // any other byte (ASCII 0x... or leading 11...) starts a new code point, so reset the buffer
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // while the next byte is a continuation byte, continue appending
        // (at most 4 bytes, to avoid overrunning str_buffer)
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // c+1 is not a continuation byte, so we've read in a full code point
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // BPE: at each iteration, merge the consecutive pair with the highest score in vocab_scores
    while (1) {
        fp16 best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab: record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}