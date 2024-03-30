#include "bert.h"
#include "ggml.h"

#include "unistd.h"
#include <stdio.h>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bert_params params;
    params.model = "models/bge-small-en-v1.5/ggml-model-q4_0.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert_ctx * bctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if ((bctx = bert_load_from_file(params.model)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int N = bert_n_max_tokens(bctx);
    // tokenize the prompt

    std::vector<const char *> texts = {
        (const char *)"你好世界",
        (const char *)"こんにちは、世界！",
        (const char *)"hello world",
    };

    const int32_t d_emb = bert_n_embd(bctx);
    
    std::vector<float> embeddings_buff(texts.size() * d_emb);
    std::vector<float *> embeddings_batch(texts.size());
    for (int i = 0; i < texts.size(); i++) {
        embeddings_batch[i] = embeddings_buff.data() + i * d_emb;
    }

    int test_batch_size = texts.size();

    int64_t t_eval_us  = 0;
    int64_t t_start_us = ggml_time_us();

    bert_encode_batch(bctx, params.n_threads, test_batch_size, test_batch_size, texts.data(), embeddings_batch.data());

    t_eval_us += ggml_time_us() - t_start_us;

    for (int i = 0; i < texts.size(); i++) {
        printf("[");
        for (int j = 0; j < 10; j++) {
            printf("%1.9f, ", embeddings_batch[i][j]);
        }
        printf("]\n");
    }


    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        //printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:  eval time = %8.2f ms / %.2f ms per input\n", __func__, t_eval_us/1000.0f, t_eval_us/1000.0f/texts.size());
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}