#include "MHSA.h"


void llama2_mhsa_fp16_cl(void *llama2_mhsa_args){
    struct llama2_mhsa_args_fp16* args = (struct llama2_mhsa_args_fp16*) llama2_mhsa_args;

    int pos = args->pos;
    int kv_dim = args->kv_dim;
    int kv_mul = args->kv_mul;
    int head_size = args->head_size;
    int n_heads = args->n_heads;
    int steps = args->steps;
    const fp16 sqrt_head_size = (fp16) sqrtf(head_size);

    const uint32_t blockSize = (n_heads + NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > n_heads ? n_heads : start+blockSize;

    for (int h = start; h < stop; h++) {
            // get the query vector for this head
            fp16* q = args->q + h * head_size;
            // attention scores for this head
            fp16* att = args->att + h * steps;
            // iterate over all timesteps, including the current one

            fp16 max_val = -65504;    // lowest finite fp16 value
            int t;
            for(t=0; t <= pos-3; t+=4) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp1 = (v2f16) {0, 0};
                v2f16 temp2 = (v2f16) {0, 0};
                v2f16 temp3 = (v2f16) {0, 0};
                v2f16 temp4 = (v2f16) {0, 0};

                v2f16 A, B1, B2, B3, B4;

                for (int i = 0; i < head_size; i+=2) { //+=2 because each v2f16 vector holds 2 elements
                    A = *((v2f16*) &q[i]);
                    
                    B1 = *((v2f16*) &k[i]);
                    B2 = *((v2f16*) &k[i + kv_dim]);
                    B3 = *((v2f16*) &k[i + 2*kv_dim]);
                    B4 = *((v2f16*) &k[i + 3*kv_dim]);

                    /**
                     * TASK 1.3.
                     *
                     * Accumulate the dot products between the query q and the keys of the 4 timesteps t..t+3,
                     * along the head_size dimension. The loop is unrolled over 4 keys, so that each element of
                     * the query (A) is loaded once and reused for all 4 keys.
                     * 
                     *  - A:      elements i and i+1 of the query
                     *  - B1..B4: elements i and i+1 of the keys at timesteps t, t+1, t+2, t+3
                     *  - temp1..temp4: one accumulator per key
                     * 
                     * Arithmetic operators on v2f16 work element-wise on both lanes: each accumulator keeps
                     * two partial sums, which are added together after the loop (see att[t] below).
                     */
                }

                // save the score to the attention buffer
                att[t] = (temp1[0] + temp1[1]) / sqrt_head_size;
                if(att[t] > max_val) 
                    max_val = att[t];
                
                att[t+1] = (temp2[0] + temp2[1]) / sqrt_head_size;
                if(att[t+1] > max_val)
                    max_val = att[t+1];
                
                att[t+2] = (temp3[0] + temp3[1]) / sqrt_head_size;
                if(att[t+2] > max_val)
                    max_val = att[t+2];
                
                att[t+3] = (temp4[0] + temp4[1]) / sqrt_head_size;
                if(att[t+3] > max_val)
                    max_val = att[t+3];
            }
            
            // leftover
            while(t <= pos) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp = (v2f16) {0, 0};
                v2f16 A,B;

                for (int i = 0; i < head_size; i+=2) {
                    A = *((v2f16*) &q[i]);
                    B = *((v2f16*) &k[i]);

                    /**
                     * TASK 1.3.
                     *
                     * Same as above, for a single key: this loop handles the last timesteps when the
                     * number of keys (pos+1) is not a multiple of 4.
                     * Accumulate the dot product between the query (A) and the key at timestep t (B) into temp.
                     */
                }
                // save the score to the attention buffer
                att[t] = ( temp[0] + temp[1] ) / sqrt_head_size;
                if(att[t] > max_val)
                    max_val = att[t];
                t++;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            // softmax_original_fp16(att, pos + 1);
            fp16 sum = 0.0f;
            for (int t = 0; t < pos+1; t++) {
                // FastExp
                float x = (float) (att[t] - max_val);
                x = GIST_A * x + GIST_B;
                if (x < GIST_C )    // no need to check if x > GIST_D, because x <= 0
                    x = 0.0f;

                uint32_t n = (uint32_t) (x);
                att[t] = (fp16) *(float*) &n;

                sum += att[t];
            }

            // weighted sum of the values, store back into xb
            fp16* xb = args->xb + h * head_size;
            fp16* v = args->value_cache + (h / kv_mul) * head_size;

            // for each t:  xb += v[t] * att[t];
            for(int i=0 ; i < head_size ; i+=2){           
                v2f16 temp = (v2f16) {0, 0};
                for(int t = 0; t <= pos; t++){
                    temp += *((v2f16*)&v[i + t*kv_dim]) * (v2f16) {att[t], att[t]};
                }
                xb[i] = temp[0] / sum;
                xb[i+1] = temp[1] / sum;
            }
    }
}