#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// OPTIONS
#define NUM_CORES 8
#define PROMPT "Once upon a time"
#define STEPS 256
#define TEMPERATURE 1.0f
#define RND_SEED 42

#define MODEL_PATH "Model/stories260K.bin"
#define TOKENIZER_PATH "Model/tokenizer.bin"
#define HEADER_FILE "conf_and_weights_fp16.h"
#define SOURCE_FILE "conf_and_weights_fp16.c"


// Structures used in the model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int max(int a, int b){
    return a>b? a : b;
}

int main(void){

    FILE* fw = fopen(MODEL_PATH, "rb");
    if(fw == NULL){
        printf("Error: can't open file: %s\n\n", MODEL_PATH);
        exit(1);
    }

    Config c;
    if(fread(&c, sizeof(Config), 1, fw)!=1){
        fprintf(stderr, "Error: unexpected format of %s\n", MODEL_PATH);
        exit(1);
    }
    fseek(fw, 0, SEEK_END);
    int file_size = ftell(fw);
    fclose(fw);

    int fwd = open(MODEL_PATH, O_RDONLY);
    float* data;
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fwd, 0);
    if(data == MAP_FAILED){
        printf("Error: mmap failed\n");
        exit(1);
    }

    int head_size = c.dim / c.n_heads;
    float* w = data + sizeof(Config)/sizeof(float);
    int w_dim = c.dim * (c.vocab_size + c.n_layers*(2 + 2*c.dim + head_size*c.n_kv_heads*2 + 3*c.hidden_dim) + 1);

    // Override c.seq_len with STEPS to reduce memory usage
    c.seq_len = STEPS;

    // Read tokenizer
    Tokenizer t;
    FILE *file_tok = fopen(TOKENIZER_PATH, "rb");
    if(file_tok == NULL){
        printf("Error: can't open file: %s\n", TOKENIZER_PATH);
        exit(1);
    }
    t.vocab_size = c.vocab_size;
    t.vocab = (char**) malloc(c.vocab_size*sizeof(char*));
    t.vocab_scores = (float*) malloc(c.vocab_size*sizeof(float));

    if(fread(&t.max_token_length, sizeof(int), 1, file_tok)!=1){
        printf("Error while reading tokenizer file (%s)\n", TOKENIZER_PATH);
        exit(1);
    }

    int len, tot_vocab_size=0;
    for(int i=0; i<c.vocab_size; i++){
        if(fread(t.vocab_scores+i, sizeof(float), 1, file_tok)!=1){
            printf("Error while reading tokenizer file (%s)\n", TOKENIZER_PATH);
            exit(1);
        }
        if(fread(&len, sizeof(int), 1, file_tok)!=1){
            printf("Error while reading tokenizer file (%s)\n", TOKENIZER_PATH);
            exit(1);
        }
        t.vocab[i] = (char *)malloc(len+1);
        tot_vocab_size += len+1;
        if(fread(t.vocab[i], sizeof(char), len, file_tok)!=(size_t)len){
            printf("Error while reading tokenizer file (%s)\n", TOKENIZER_PATH);
            exit(1);
        }
        t.vocab[i][len] = '\0';
    }
    fclose(file_tok);

    t.sorted_vocab = malloc(t.vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t.vocab_size; i++) {
        t.sorted_vocab[i].str = t.vocab[i];
        t.sorted_vocab[i].id = i;
    }
    qsort(t.sorted_vocab, t.vocab_size, sizeof(TokenIndex), compare_tokens);

    // Header file: configuration and declarations
    FILE* fh = fopen(HEADER_FILE, "w");
    if(fh==NULL){
        printf("Error: can't create file: %s\n", HEADER_FILE);
        exit(1);
    }

    fprintf(fh, "// Config Definition\n");
    fprintf(fh, "#pragma once\n");
    fprintf(fh, "#include \"pulp_train.h\"\n\n");
    fprintf(fh, "#define DIM %d\n", c.dim);
    fprintf(fh, "#define HIDDEN_DIM %d\n\n", c.hidden_dim);
    fprintf(fh, "#define N_LAYERS %d\n", c.n_layers);
    fprintf(fh, "#define N_HEADS %d\n", c.n_heads);
    fprintf(fh, "#define N_KV_HEADS %d\n\n", c.n_kv_heads);
    fprintf(fh, "#define VOCAB_SIZE %d\n", c.vocab_size);
    fprintf(fh, "#define SEQ_LEN %d\n", c.seq_len);
    fprintf(fh, "#define KV_DIM %d\n\n", c.dim * c.n_kv_heads / c.n_heads);
    fprintf(fh, "#define STEPS %d\n", STEPS);
    fprintf(fh, "#define TEMPERATURE %.3ff\n\n", TEMPERATURE  );
    fprintf(fh, "#define RND_SEED %d\n\n", RND_SEED);

    fprintf(fh, "PI_L2 extern char* PROMPT;\n");
    fprintf(fh, "PI_L2 extern int PROMPT_TOKENS[%zu];\n\n", strlen(PROMPT)+3);

    // the L1 buffers are defined in net.c, only BUFF1 (used by the tokenizer) and buffer_n_cores (used by the sampler) are exported
    fprintf(fh, "extern PI_L1 fp16 BUFF1[DIM];\n");
    fprintf(fh, "extern PI_L1 fp16 buffer_n_cores[NUM_CORES];\n\n");

    fprintf(fh, "// RunState allocation\n");
    fprintf(fh, "PI_L2 extern fp16 KEY_CACHE [N_LAYERS*STEPS*KV_DIM];\n");
    fprintf(fh, "PI_L2 extern fp16 VALUE_CACHE [N_LAYERS*STEPS*KV_DIM];  // KV_DIM = DIM * N_KV_HEADS/ N_HEADS\n\n");
    fprintf(fh, "PI_L2 extern char PROB_INDEX [VOCAB_SIZE*%zu];\n\n", sizeof(ProbIndex));

    fprintf(fh, "// Tokenizer allocation\n");
    fprintf(fh, "#ifndef _TokenIndex_\n#define _TokenIndex_\n\ntypedef struct {\nchar *str;\nint id;\n} TokenIndex;\n#endif\n\n");
    fprintf(fh, "#define MAX_TOKEN_LENGTH %d\n\n", t.max_token_length);
    fprintf(fh, "PI_L2 extern fp16 VOCAB_SCORES [VOCAB_SIZE];\n\n");
    fprintf(fh, "PI_L2 extern char* VOCAB[VOCAB_SIZE];\n");
    fprintf(fh, "PI_L2 extern unsigned char VOCAB_DATA [%d];\n\n", tot_vocab_size);
    fprintf(fh, "PI_L2 extern TokenIndex SORTED_VOCAB [VOCAB_SIZE];\n\n");

    fprintf(fh, "// Model weights\n");
    fprintf(fh, "PI_L2 extern fp16 weights_list[%d];\n", w_dim);
    fclose(fh);
    printf("Writing %s finished\n", HEADER_FILE);

    // Source file: definitions (prompt, KV cache, tokenizer and weights)
    FILE* fc = fopen(SOURCE_FILE, "w");
    if(fc==NULL){
        printf("Error: can't create file: %s\n", SOURCE_FILE);
        exit(1);
    }

    fprintf(fc, "#include \"%s\"\n\n", HEADER_FILE);

    fprintf(fc, "PI_L2 char* PROMPT = \"%s\";\n", PROMPT);
    fprintf(fc, "PI_L2 int PROMPT_TOKENS[%zu];\n\n", strlen(PROMPT)+3);

    fprintf(fc, "// RunState allocation\n");
    fprintf(fc, "PI_L2 fp16 KEY_CACHE [N_LAYERS*STEPS*KV_DIM];\n");
    fprintf(fc, "PI_L2 fp16 VALUE_CACHE [N_LAYERS*STEPS*KV_DIM];\n");

    // aligned: Sampler.c accesses it as an array of ProbIndex
    fprintf(fc, "\nPI_L2 char PROB_INDEX [VOCAB_SIZE*%zu] __attribute__((aligned(4)));\n\n", sizeof(ProbIndex));

    fprintf(fc, "// Tokenizer allocation\n");
    fprintf(fc, "PI_L2 fp16 VOCAB_SCORES [VOCAB_SIZE] = {\n");
    int i;
    for(i=0;i<c.vocab_size-1;i++){
        fprintf(fc, "%f, ", t.vocab_scores[i]);
        if(i%10==9)
            fprintf(fc, "\n");
    }
    fprintf(fc, "%f};\n\n", t.vocab_scores[i]);

    fprintf(fc, "PI_L2 char* VOCAB[VOCAB_SIZE];\n");
    fprintf(fc, "PI_L2 unsigned char VOCAB_DATA [%d] = {\n", tot_vocab_size);
    for(i=0;i<c.vocab_size;i++){
        int j=0;
        while(t.vocab[i][j]!='\0')
            fprintf(fc, "0x%02x, ", (unsigned char) t.vocab[i][j++]);
        if(i<c.vocab_size-1)
            fprintf(fc, "0x%02x, \n", (unsigned char)'\0');
        else
            fprintf(fc, "0x%02x };\n\n", (unsigned char)'\0');
    }


    fprintf(fc, "PI_L2 TokenIndex SORTED_VOCAB [VOCAB_SIZE] = {\n");
    for(i=0; i<t.vocab_size; i++){
        fprintf(fc, "{\"");
        for(int k=0; t.sorted_vocab[i].str[k] != '\0'; k++)
            fprintf(fc, "\\x%02X", (unsigned char) t.sorted_vocab[i].str[k]);
        if(i< t.vocab_size-1)
            fprintf(fc, "\", %d},\n", t.sorted_vocab[i].id);
        else
            fprintf(fc, "\", %d}\n", t.sorted_vocab[i].id);
    }
    fprintf(fc, "};\n\n");


    fprintf(fc, "// Model weights\n");
    fprintf(fc, "PI_L2 fp16 weights_list[%d] = { ", w_dim);
    for(i=0;i<w_dim-1;i++){
        fprintf(fc, "%.10f, ", w[i]);
        if(i%10 == 9)
            fprintf(fc, "\n");
    }
    fprintf(fc, "%.10f};", w[i]);
    fclose(fc);
    printf("Writing %s finished\n", SOURCE_FILE);

    // L1 buffers defined in net.c: they must be at least this large
    int buff1_dim = c.dim;
    int buff2_dim = c.dim;
    int buff3_dim = max(c.dim, c.hidden_dim);
    int buff4_dim = max(max(c.dim, c.vocab_size), max(c.n_heads*STEPS, c.hidden_dim));
    int buff_w_dim = max(max(c.dim*c.dim, STEPS*c.dim*c.n_kv_heads/c.n_heads), c.dim*c.hidden_dim);

    printf("\nMinimum L1 buffer sizes (see net.c): BUFF1[%d] BUFF2[%d] BUFF3[%d] BUFF4[%d] BUFF_W_1[%d] BUFF_W_2[%d]\n",
           buff1_dim, buff2_dim, buff3_dim, buff4_dim, buff_w_dim, buff_w_dim);

    const int sizeof_fp16 = 2;

    int tot_L1 = buff1_dim + buff2_dim + buff3_dim + buff4_dim + 2*buff_w_dim + NUM_CORES;
    tot_L1 *= sizeof_fp16;

    int tot_L2 = (2*c.n_layers * STEPS * c.dim * c.n_kv_heads / c.n_heads) * sizeof_fp16; // KV_CACHE
    tot_L2 += c.vocab_size * (sizeof(int) + sizeof(float)); // PROB_INDEX
    tot_L2 += c.vocab_size * sizeof_fp16; // VOCAB_SCORES
    tot_L2 += c.vocab_size * sizeof(char*); // VOCAB
    tot_L2 += tot_vocab_size * sizeof(char); // VOCAB_DATA
    tot_L2 += tot_vocab_size * sizeof(char) + c.vocab_size * (sizeof(char*) + sizeof(int)); // SORTED_VOCAB
    tot_L2 += w_dim * sizeof_fp16;

    printf("\nL1 memory usage: %d Byte (%g kB)\n", tot_L1, tot_L1/1024.0f);
    printf("L2 memory usage: %d Byte (%g kB)\n", tot_L2, tot_L2/1024.0f);

    return 0;
}
