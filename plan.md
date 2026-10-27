# MuJoCo HalfCheetah — Semester Project Plan

This is a living checklist. 

## 1. Checkpoint 1 — 29 October 2026: environment and proposal

### Environment setup and exploration

- [x] Create the repository, Python package structure, `.gitignore` and dependency configuration.
- [x] Install Gymnasium with MuJoCo support; successfully run `HalfCheetah-v5` locally.
- [x] Record exact OS, Python/library versions and hardware details.
- [x] Inspect observation/action spaces and verify shapes, dtypes, bounds and action order.
- [x] Explain the **agent objective**, physical model and direction of travel.
- [x] Derive the default reward equation and identify forward reward and control cost separately.
- [x] Verify termination versus time-limit truncation and the default episode horizon.
- [x] Inspect configurable environment parameters and decide which are actually worth investigating.
- [x] Confirm useful `info` fields and that position, velocity and reward components can be logged.
- [x] Add a reproducible environment smoke test and a seeded random-policy rollout.

### Required random-agent video

- [x] Implement random actions sampled from `env.action_space` on each step, without learning or heuristics.
- [x] Record a short video using `render_mode="rgb_array"` and a suitable Gymnasium recording wrapper or video encoder.
- [x] Validate playback, visible movement, duration and a clear “Random policy” label.
- [x] Save the video and note the seed, environment version, recording settings and command.

### Research and presentation

- [ ] Confirm the course-assigned research question and difficulty color.
- [ ] Write the provisional main hypothesis, experiment variables and controls.
- [x] Choose a random-agent baseline and a trained SAC baseline; state whether PPO will be an optional comparison.
- [x] Define intermediate goal: a trained policy clearly outperforming random on standard settings.
- [x] Define final goal: reproducible trained policy and evidence-based answer to the research question; avoid promising an unverified return threshold.
- [x] Define evaluation measures: return, forward speed, displacement, mean action-squared, reward components, episode length and across-seed variability.
- [ ] Create a **Polish-language** checkpoint slide deck covering all four requested sections: environment/task; configuration; random-agent video; further work plan.
- [ ] Rehearse the presentation and verify video works offline during the presentation.
- [ ] Attend scheduled in-person consultation with the team; record feedback and revise the plan.

**Acceptance:** environment launches; random-agent video exists; slides accurately explain reward, observations and actions; research scope, baseline and metrics are explicit.

## 2. Checkpoint 2 — 3 December 2026: working agent

- [ ] Implement a configuration-driven environment factory and deterministic seeding.
- [ ] Implement episode metrics and structured logging (CSV/TensorBoard; preserve all runs).
- [ ] Implement training, checkpoint saving, resume and evaluation entry points.
- [ ] Train the first SAC policy; confirm it improves materially over the random baseline under matched conditions.
- [ ] Separate training runs from evaluation runs; evaluate full episodes without training exploration noise.
- [ ] Validate that model loading reproduces evaluation behavior.
- [ ] Run at least a small seed repeat (ideally 3) and plot learning curves with uncertainty.
- [ ] Diagnose any instability, reward hacking, excessive action magnitudes or poor forward progress.
- [ ] Produce a short demo of the trained agent and compare with random policy.
- [ ] Finalize the controlled experiment grid based on early results and compute availability.
- [ ] Prepare **Polish-language** progress slides with functioning agent, initial metrics and remaining work.
- [ ] Attend consultation, capture criticism and make an updated experimental plan.

**Acceptance:** reproducible working trained agent, checkpointed model, preliminary comparison and an actionable research plan.

## 3. Research phase — December 2026 to January 2027

### Experimental design

- [ ] Write a protocol stating question, hypothesis, independent variable, controls, budgets, seed policy and evaluation metrics.
- [ ] Run a reference SAC configuration with fixed baseline environment settings.
- [ ] If the assigned question allows it, sweep `ctrl_cost_weight` over `0.0`, `0.1`, `0.5` with equal training steps and model architecture.
- [ ] Save all run configs, seeds, checkpoints, learning curves, wall-clock time and raw evaluation results.
- [ ] Evaluate at the same checkpoints using a fixed, held-out evaluation seed list.
- [ ] Compare speed versus action energy proxy; **do not directly compare raw rewards when reward weights differ**.
- [ ] Analyze variance, non-convergence and cases where the hypothesized effect is absent.
- [ ] Formulate **at least one new hypothesis based on an earlier experiment**; run a controlled follow-up test of that hypothesis.
- [ ] Include negative/inconclusive results and plausible alternative explanations.
- [ ] Optional (time permitting): compare PPO with SAC, investigate observation inclusion, or perform another parameter ablation relevant to the official question.

### Scope adjustment by difficulty rating

- [ ] **Green:** target a strong working agent and multiple meaningful parameter variants/ablations.
- [ ] **Yellow:** prioritize a reliable working agent; narrow to a well-justified subset of experiments if necessary.
- [ ] **Red:** define a bounded question and evaluate an agent good enough to test it, even if not state-of-the-art.
- [ ] Record the exact selected scope and justify deviations in the report.

### Testing and reproducibility

- [ ] Unit-test action/observation assumptions, reward-component extraction and metric aggregation.
- [ ] Smoke-test short training/evaluation loops and video recording.
- [ ] Run an end-to-end test from config to model to evaluation CSV and plots.
- [ ] Document exact reproducibility commands and dependencies.
- [ ] Ensure generated media, logs and heavy model weights are organized and not accidentally committed.

## 4. Final checkpoint — 21 January 2027

- [ ] Obtain the **instructor-provided test set** when released; document its evaluation rules.
- [ ] Run final model evaluation on that test set and include results in the final work.
- [ ] Produce publication-quality figures: training curves, baseline comparisons and research-variable effects.
- [ ] Summarize the experimental methodology, results, variability and limitations.
- [ ] Answer the official research question directly, including conditions where the effect appears/disappears.
- [ ] Describe and interpret the follow-up hypothesis experiment.
- [ ] Include own reasoning and ideas, not only screenshots or generated plots.
- [ ] Include an **AI usage disclosure**: tool names, tasks, extent of use and human verification.
- [ ] Prepare the final **English** report with figures, setup details, results and references.
- [ ] Review/annotate code and ensure installation and core commands work on a clean checkout.
- [ ] Prepare and rehearse the final **Polish-language** presentation and trained-agent demo.
- [ ] Ensure required in-person team attendance at the final presentation and consultations.
- [ ] Submit all required code, report and presentation materials by the announced procedure.

**Acceptance:** commented code, reproducible experiments with logs, evidenced research answer, mandatory test-set results, final report and Polish presentation.

## 5. Definition of success

- **Minimum:** environment runs; random and trained agents are demonstrable; sound evaluation methodology and required deliverables are complete.
- **Intermediate:** SAC reliably beats random on held-out episodes and produces meaningful motion.
- **Final:** controlled experiments answer the assigned question with variance estimates, independent physical metrics, a follow-up hypothesis test and a defensible discussion.
