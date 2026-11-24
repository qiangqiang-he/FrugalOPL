# FrugalOPL
A lightweight framework for efficient on-policy learning in LLMs.

TCPO_v1 与 TCA 的 Qwen3-1.7B / 4B 四份服务器配置、16×16 batch、组内加权目标、32K AMC23 / AIME24 Avg@16 评测和启动命令见 [训练说明](docs/tcpo_v1.md)。

Qwen3-4B-Instruct-2507、Olmo 3 7B Instruct、Gemma 4 E4B IT（No thinking）与 Phi 4 Mini Instruct 的 **8 卡 TCA** 配置使用 **32 题 × 8 回复**，100 steps，Top-k 512，训练 8192 / 评测 32768；启动命令及各模型 LoRA 适配见 [8 卡 Instruct TCA](docs/tca_instruct_8gpu.md)。

`GRPOPlugin` is the shared binary-reward RL baseline for both training and
Controller/Actor/Learner integration checks. It renders prompts, obtains verified
rewards (with answer probing for truncated responses), computes group-relative
advantages, and applies the clipped policy objective without a Teacher Worker.
Its diagnostics include advantage mean/std, nonzero-advantage fraction, and policy
log probability under the `grpo_*` metric prefix. The standard GRPO advantage
normalization, clipping settings, optional KL, and probe threshold are shared
across these uses.

`configs/common/shared_batched/` and `configs/common/disjoint_async/` each contain
`2GPU.yaml`, `4GPU.yaml`, and `8GPU.yaml`. All six configurations are complete and
self-contained: they do not use `extends` or reference another configuration to
fill in missing settings. Training and evaluation entry points may extend a
common configuration and override their experiment settings. These hardware
profiles contain no algorithm, model path, dataset path, training rollout count,
or total step count; experiments supply those inputs. Evaluation presets belong
under `configs/eval/`.

| Profile | Actor GPUs | Learner GPUs | `controller.max_inflight_batches` |
| --- | --- | --- | --- |
| `shared_batched/2GPU.yaml` | 0–1 | 0–1 | 4 |
| `shared_batched/4GPU.yaml` | 0–3 | 0–3 | 4 |
| `shared_batched/8GPU.yaml` | 0–7 | 0–7 | 4 |
| `disjoint_async/2GPU.yaml` | 0 | 1 | 4 |
| `disjoint_async/4GPU.yaml` | 0–1 | 2–3 | 4 |
| `disjoint_async/8GPU.yaml` | 0–3 | 4–7 | 4 |

GPU lists must be disjoint or identical. With an in-flight limit of `K`,
disjoint Workers overlap generation and learning and generate
`max(1, K // 2)` global batches per Actor wave. Identical lists generate `K`
global batches, then consume each batch exactly once before handing the GPUs
back to the Actors; a window of one is serial by default. The final wave uses
only the remaining batches (for example, ten steps with `K=4` use `4+4+2`
in shared mode). A global batch is sharded by question across Actor GPUs,
keeping all rollouts for each question together. Partial
overlap is rejected. Actor and Learner no longer accept separate
`max_batches_per_turn` settings: change only `controller.max_inflight_batches`.
For same-card execution with one in-flight batch,
`controller.overlap_actor_learner: true` enables per-card handoff and a shared
one-rollout Learner queue, followed by one synchronized optimizer update.
`actor.evaluation.distribute_replies: true` spreads each question's replies
across Actors in both periodic and standalone evaluation. See
[Actor/Learner overlap](docs/actor_learner_overlap.md) for configuration and validation.
Teacher forwarding retains its existing independent budget. Strictly serial
validation uses an explicit window of one in the validation entry point.

The three scheduling smoke configurations under `tests/configs/` each request
ten GRPO updates over DAPO-17K, with 32 questions and eight rollouts per update.
Their model and dataset paths are specific to the selected validation server.
On the dual RTX 4090, launch training with `bash scripts/start_train.sh --tmux <config.yaml>`
(tmux is also the default). On the user-operated eight-GPU server, use
`bash scripts/start_train.sh --no-tmux <config.yaml>` to exec Python in the foreground.
`scripts/start_eval.sh` accepts the same mode flags. Foreground mode preserves
the Python PID, signals and exit code, mirrors output to the launcher log, and
does not create a tmux session or `.status` file. The detached
tmux command explicitly carries `GPU_OWNER`, `GPU_TASK`, `GPU_HOURS`, `GPU_KIND`,
`GPU_LOG`, and an optional `GPU_RUN` into Python. `GPU_LOG` defaults to the actual
launcher log; test configurations keep their logs and outputs in `tests/outputs/`.
Use `FRUGALOPL_SWANLAB_ENV_FILE` to load a readable credential file outside the
repository inside the detached training shell. Only its path is passed to tmux.

Controller evaluations use all configured Actor GPUs. Questions are divided
between GPUs, with at most one outstanding request and 64 rollouts per request
on each GPU. Returned batches are checked against their requested GPU and exact
policy version, then restored to dataset order before computing metrics.
Evaluation requests use separate queues and do not consume training in-flight
slots. After evaluation, shared GPUs enter the next Actor generation round
before the Learner can train.

Training YAML configures evaluation and persistent training checkpoints independently:

```yaml
eval_interval: 100
save_ckpt_interval: 100
```

Both settings are top-level non-negative integers measured in training steps.
Set either to `0` to disable its periodic action. `eval_interval` controls
evaluation; `save_ckpt_interval` saves a complete LoRA training checkpoint under
`<output_root>/<run.name>/checkpoints/step_<N>`, including the adapter,
optimizer, scheduler (when configured), RNG state, and plugin training state.
Saving does not depend on evaluation being
enabled or on its interval. Periodic checkpoints are retained;
`learner.checkpoint.max_keep` limits
only temporary publications used for Actor weight updates and exact-step
recovery. Evaluation adapters remain under `eval_adapters/`.
Temporary publications retain at least `2 * controller.max_inflight_batches + 2`
versions, even when `max_keep` is smaller, so queued Actor waves can load their
exact sampling adapter. A larger configured retention is respected.
The periodic checkpoint implementation requires LoRA to be enabled. On restart,
unconfirmed temporary and periodic checkpoints are preserved under `recovery/`
before those training steps are regenerated.

To record per-question training accuracy, enable one top-level switch:

```yaml
save_question_accuracy: true  # default: false
```

Each completed batch saves `question_accuracy/train-<batch_number>.json` under
the run output directory. It contains only an ordered array of
`{"question_index": 123, "accuracy": 0.75}` entries. `question_index` is the
zero-based position in the original training JSON record array, before shuffle.
Accuracy is the fraction of that question's generated rollouts whose final
boxed answer passes the existing evaluation verifier. It uses the actual
rollout count and preserves separate batch positions for repeated questions.
These diagnostics do not enter the algorithm's rewards, loss, or experiment
metrics. When disabled, no per-question accuracy is computed or saved.

All algorithms share the framework's loss aggregation setting, configured by
each experiment under `algorithm` (the default is `sequence`):

```yaml
algorithm:
  loss_aggregation: sequence  # sequence | token
```

`sequence` averages each response's valid token objectives, then averages all
responses equally. `token` divides the sum of objectives over all valid tokens
by the total valid token count across the complete logical batch and all Learner
ranks. Each objective must report its actual loss mask's token count; generated
length, padding, or non-finite excluded tokens cannot substitute for it. GRPO's
policy surrogate and optional KL use the same aggregation. Gradient accumulation
is normalized before gradient clipping, and loss telemetry uses the same weights.
An older custom plugin without a token count can still use `sequence`; choosing
`token` raises an explicit error until the plugin supplies that count.
GRPO presets explicitly use `sequence`, which is also the default for future
GRPO configurations.

GRPO configures its lower and upper clipping margins separately:

```yaml
algorithm:
  epsilon_low: 0.1
  epsilon_high: 0.2
```

The policy surrogate is `min(ratio * advantage, clamp(ratio, 1 - epsilon_low,
1 + epsilon_high) * advantage)`. The example clips the ratio to `[0.9, 1.2]`;
both defaults are `0.2`, preserving the former symmetric `[0.8, 1.2]` behavior.
`epsilon_low` must be finite and in `(0, 1)`, while `epsilon_high` must be finite
and positive. The former GRPO `clip_ratio` key is rejected; use the two keys
directly under `algorithm`.

GRPO can scale positive and negative group-relative advantages separately:

```yaml
algorithm:
  positive_advantage_scale: 1.0
  negative_advantage_scale: 0.6
```

Each rollout's advantage is first computed using the standard GRPO group mean
and sample standard deviation, `(reward - group_mean) / (group_std + 1e-4)`.
The corresponding scale is then applied once according to the advantage's sign,
before the clipped policy surrogate. There is no subsequent centering or
renormalization. Rewards and the optional KL term are not scaled. Both keys
default to `1.0` when omitted, and accept only finite, non-negative numbers;
`0.0` is allowed.

GRPO can also select an entropy-driven dynamic advantage strategy:

```yaml
algorithm:
  positive_advantage_scale: 1.0
  negative_advantage_scale: 1.0
  dynamic_advantage:
    enabled: true             # default: false
    baseline_steps: 100
    window_steps: 10
    min_negative_scale: 0.1    # allowed range: [0.1, 1.0]
```

Enabling this strategy requires both static advantage scales to be `1.0`.
It leaves positive advantages unchanged and controls only the magnitude of
negative advantages after the usual group standardization. With the switch
disabled or omitted, the existing static scales apply unchanged.
`min_negative_scale` must be finite and in `[0.1, 1.0]`; it floors every
selected coefficient.

The first 100 completed logical training batches use standard GRPO. Their
globally aggregated `grpo_entropy` values form a running mean, which is frozen
after step 100 as the baseline entropy for the rest of the run. Evaluation
requests never contribute to either this baseline or the training windows.
There is no low-entropy adjustment: ratios below `0.25` also retain a negative
advantage scale of `1.0`.

Each subsequent non-overlapping 10-step window calculates its mean training
entropy. The ratio of this window mean to the fixed baseline selects the next
window's negative advantage scale:

| Window entropy / baseline entropy | Negative advantage scale |
| --- | --- |
| `r <= 1.5` | `1.0` |
| `1.5 < r <= 2` | `0.9` |
| `2 < r <= 3` | `0.8` |
| `3 < r <= 4` | `0.7` |
| `4 < r <= 5` | `0.6` |
| `5 < r <= 6` | `0.5` |
| `6 < r <= 7` | `0.4` |
| `7 < r <= 8` | `0.3` |
| `8 < r <= 9` | `0.2` |
| `r > 9` | `0.1` with the shown minimum |

Steps 101–110 therefore still use `1.0`. Their mean entropy determines the
coefficient used by steps 111–120; steps 111–120 then determine the coefficient
used by steps 121–130, and so on. Every decision is calculated
again from the fixed baseline; coefficients do not multiply cumulatively and
can return to `1.0` when the ratio falls to 1.5 or below. Decisions apply at
logical-batch boundaries, including boundaries inside a shared-GPU `K=4`
round. Scaling does not change how often a batch is trained.

SwanLab records the following scalars under the `grpo/` namespace:

| Metric | Meaning |
| --- | --- |
| `dynamic_advantage_enabled` | Whether the strategy is enabled |
| `dynamic_advantage_alpha_neg` | Coefficient actually used for this step |
| `dynamic_advantage_alpha_neg_next` | Coefficient for the next training step |
| `dynamic_advantage_baseline_entropy` | Running baseline mean, then the frozen baseline |
| `dynamic_advantage_baseline_count`, `dynamic_advantage_baseline_ready` | Baseline collection progress |
| `dynamic_advantage_window_count`, `dynamic_advantage_window_running_entropy` | Current training window progress and running mean |
| `dynamic_advantage_training_entropy`, `dynamic_advantage_entropy_ratio` | Most recently completed window mean and its ratio to the baseline |

The completed-window metrics first appear at step 110, and that decision first
applies at step 111. At a decision boundary,
`alpha_neg` describes the update just completed, while `alpha_neg_next`
describes the new decision. Checkpoints save the baseline accumulators,
partial window, completed step, and coefficient alongside the model and
optimizer state. Resuming a dynamic run requires matching algorithm state;
it does not silently rebuild the baseline from later training data.
Checkpoints must also match the strategy revision: changing the threshold rule
or window length requires a fresh run rather than reusing older dynamic state.

`configs/training/grpo/` contains six Qwen3-1.7B training entry points using
standard group-mean GRPO advantages (`advantage_estimator: grpo`):

| Scheduling | 2 GPUs | 4 GPUs | 8 GPUs |
| --- | --- | --- | --- |
| Shared, batched | `shared_batched_2GPU.yaml` | `shared_batched_4GPU.yaml` | `shared_batched_8GPU.yaml` |
| Disjoint, asynchronous | `disjoint_async_2GPU.yaml` | `disjoint_async_4GPU.yaml` | `disjoint_async_8GPU.yaml` |

Each entry point inherits the matching complete common hardware configuration.
They use the same training settings: DAPO-17K, LoRA rank `64` / alpha `128`,
1000 steps, and 32 questions with eight rollouts each per update. Each global
batch contains 256 rollouts and is consumed once. The in-flight limit is `K=4`;
training generation is limited to 8192 tokens, and clipping bounds are
`[0.9, 1.2]` (`epsilon_low=0.1`, `epsilon_high=0.2`). Training temperature
and top-p are both `1.0`. Advantages use each group's mean reward baseline
and original reward standard deviation. Both static advantage scales are
fixed at `1.0`; dynamic negative-advantage scaling is disabled.

QAE remains available through `advantage_estimator: qae` and
`qae_quantile: 0.4`. It uses the non-interpolating empirical quantile while
preserving the reward standard deviation and full batch denominators. With
eight binary rewards, four or fewer successes give baseline zero and five or
more give baseline one.

These GRPO configurations explicitly set a constant learning rate of `1e-6`.
Evaluation uses temperature `0.6` and top-p `0.95` independently of training.
AMC23 evaluation runs before training and every 100 steps, with eight responses
per question and a 16384-token generation limit; checkpoints are saved every
50 steps. Each entry point has its own run name, output directory, and Redis
prefix.

For example, launch the shared eight-GPU configuration with
`bash scripts/start_train.sh --no-tmux configs/training/grpo/shared_batched_8GPU.yaml`.

`configs/training/pg_opsd/shared_batched_4GPU.yaml` deploys Actor and Learner
on the same four GPUs with `K=4`. It retains the original PG-OPSD Qwen3-1.7B
settings: OpenThoughts Math30K, LoRA rank `64` / alpha `128`, constant learning
rate `1e-5`, 100 updates, and 32 questions with four responses each per update.
Training generation remains 4096 tokens; initial AIME24 and AMC23 evaluation
uses eight responses per question and runs again every 20 steps, alongside
checkpoint saving. The fixed solution-conditioned Teacher forward runs inside
each Learner with LoRA disabled; there is no separate Teacher Worker. The common
profile supplies the 40 GB per-GPU Actor budget and CUDA Graph execution.

All distribution losses use Student-selected top-k support. Set
`algorithm.kl_top_k` to a positive integer; the two-, four- and eight-GPU
profiles default to **512**. Each response position selects its own Student
top-k IDs. Teacher and Student normalize on these same IDs, and Ratio/TCA
retain their plausible-support filter inside the selected set. This is a
conditional top-k objective, distinct from full-vocabulary KL.
The physical training micro-batch remains **one rollout**. The complete
response uses one head/loss checkpoint, without token chunking; the old
`kl_chunk_size`, `tail_sup_chunk_size` and `forward_chunk_size` settings are
removed from configuration. Model output logits are still needed to find
exact top-k; FP32 loss distributions and their targets are k-wide. Actual
sampled-action probabilities used by policy gradients and RADO importance
weights retain the original sampling normalization, including actions
outside top-k. Checkpoints record the support and k, and reject mismatched
objective settings on resume.
RADO's direction score is centered within the selected support and zero for
sampled actions outside it, preserving the support's total probability mass.

`KLOPSDPlugin` implements OPSD Forward KL on this top-k support with a
frozen, solution-conditioned base Teacher. Its separate four-GPU entry point,
`configs/training/kl_opsd/shared_batched_4GPU.yaml`, follows the authors' 8B
non-thinking recipe with token reduction and fresh `K=1` sampling. See
[the KL-OPSD implementation notes](docs/kl_opsd.md) for the pinned upstream
sources, pointwise clipping semantics, hyperparameters, and local validation.

`TailSupPlugin` (`tail_sup`) implements tail suppression within Student top-k using
the frozen student policy that generated each rollout. It sorts that policy's
conditional temperature-scaled distribution, includes the token crossing
`algorithm.tail_sup_top_p`, and renormalizes the head into a detached target.
Forward KL uses the current student's conditional top-k softmax without KL
clipping or a temperature-squared multiplier. Positions whose old maximum
probability strictly exceeds the head threshold are excluded from both the
loss and its token denominator. Training uses no teacher, reward, solution or
correctness label. `teacher_sync_interval: 1` provisions the internal versioned
old-student LoRA adapter, selected by the rollout's `policy_version`.

The configurations under `configs/training/tail_sup/` use Qwen3-1.7B,
100 steps, 64 questions with four rollouts each, a 4096-token response limit,
LoRA rank 64 / alpha 128, and learning rate `5e-6`. Sampling temperature and
sampling top-p are both `1.0` with top-k disabled; the independent head
threshold is `0.9`, `0.95` or `0.99`. The shared four-GPU recipe uses one fresh batch
per update. Run `bash scripts/start_tail_sup_rtx5000_1.sh` to launch the
top-p=0.9 experiment on physical GPUs 0-3 after local validation and deployment.
`bash scripts/start_tail_sup_p099_rtx5000_1.sh` starts top-p=0.99 on GPUs 4-7.
The CPU-only `scripts/wait_tail_sup_p099_rtx5000_1.sh` can check those four
cards every two seconds and launch that training as soon as they are all free.

An algorithm objective returns
`ObjectiveResult(objective, metrics, valid_token_count=valid_mask.sum().detach())`.
The objective is a maximized mean over its final valid token mask; multiplying it
by the count must recover the complete token objective sum. Learner owns the
minimization sign, cross-rollout/rank normalization, and gradient clipping. A
logical batch with no valid tokens in `token` mode skips the optimizer,
scheduler and policy-version update. Other diagnostics retain their existing
response-level means.

Stage-zero FP16 scaling is applied exactly once. DeepSpeed 0.19.7's modern
backward path requires an explicit scale before engine backward; older engines
that delegate to optimizer backward retain their native scaling. An unverified
modern scaling API explicitly fails instead of silently changing the update.
