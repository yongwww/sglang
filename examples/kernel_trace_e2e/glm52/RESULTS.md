# GLM-5.2 DSA cache-update E2E results

These results compare stock SGLang with the opt-in Kernel Factory candidate on
8x NVIDIA B300. Both sides use the same pinned InferenceX recipe, serving
image, model, TP8 configuration, CUDA-graph configuration, and request workload
documented in `README.md` and enforced by `run_exact_inferencex.sh`.

## Correctness gate

Full GSM8K (1,319 examples, 5-shot, temperature 0):

| Mode | Strict accuracy | Flexible accuracy | Server errors |
| --- | ---: | ---: | ---: |
| Stock | 96.6641% | 96.8916% | 0 |
| Candidate | 96.8916% | 97.1190% | 0 |

The candidate passed the accuracy gate. Both scores changed by +0.2275
percentage points; token-for-token equality is not claimed for distributed
serving.

Result SHA-256 values:

- stock: `5ee997f1cee41bf51ffb3621d6e34cc7467e786c562ab837b078ee3ee1ff7ad6`
- candidate: `58507044b9c55dc0151932df18135286094be9d8ddd43db0a4272afae93b7c5c`

## Exact InferenceX performance

AgentX, concurrency 8, 3,600-second measured window:

| Metric | Stock | Candidate | Candidate delta |
| --- | ---: | ---: | ---: |
| Successful requests | 1,484 | 1,430 | -3.64% |
| Request throughput | 0.41183 req/s | 0.39410 req/s | -4.31% |
| Total throughput | 40,579.71 tok/s | 38,420.60 tok/s | -5.32% |
| Input throughput | 40,182.56 tok/s | 38,055.09 tok/s | -5.29% |
| Output throughput | 397.15 tok/s | 365.51 tok/s | -7.97% |
| TTFT mean | 0.748 s | 0.748 s | +0.07% |
| TTFT p50 | 0.569 s | 0.567 s | -0.30% |
| E2E latency mean | 4.860 s | 5.738 s | +18.05% |
| E2E latency p50 | 2.272 s | 2.565 s | +12.88% |
| ITL mean | 5.23 ms | 6.31 ms | +20.65% |

Both accepted runs completed with zero profiled-request errors. The first stock
attempt encountered a transient stock cuBLAS execution failure and was excluded;
the one controlled retry completed the full window and is the stock result
above. The candidate is correct but does not improve B300 end-to-end
performance under this exact serving recipe, so it should not replace stock
based on these measurements. The candidate was generated for GB300, making this
a B300 portability/performance observation rather than a GB300 performance
claim.

Raw aggregate SHA-256 values:

- stock: `71db88f97c6a7898f5c0adee0d961976ccf08d2b18d12ac4a357ab3b4d465967`
- candidate: `14c2ce81c088f391077cde829f37cb79eee10d1e982db4856f7ff3f2295ae4d4`

The stock and candidate server-command hashes are identical:
`20b9064dd05484e9b33cf401eacf22845b9c8866dd69e226d468074b1d0441a1`.
The benchmark-command hashes are also identical:
`d80c9059485a1cc6f974f722dff45a768b8d03b48f22a9a9bd615b9f9fb402a4`.
