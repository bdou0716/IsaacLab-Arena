---
orphan: true
---

# Task success and predicates that hold for consecutive steps

Task success is defined by progress objectives. The environment builder creates
one `TaskSuccessFromProgress` termination term for those objectives. This term
owns their runtime state, advances them once per control step, and reports
completion on the step that the final required milestone completes.

The task declares success, failure, and timeout together through
`get_termination_cfg() -> TaskTerminationCfg`. Its `success` field contains the
required progress objectives, `failures` maps names to failure terms, and
`timeout_s` sets the episode time budget. The builder creates the Isaac Lab
termination terms from this configuration. `NoTask` declares no success
objectives and receives no success term.

```python
from functools import partial

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

def get_termination_cfg(self) -> TaskTerminationCfg:
    return TaskTerminationCfg(
        success=[
            ProgressObjective(
                name="insert",
                sequence=[
                    partial(
                        insertion_is_valid,
                        inserted_object_name=self.inserted_object.name,
                        receiver_name=self.receiver.name,
                    ),
                ],
            ),
        ],
        failures={},
        timeout_s=self.episode_length_s,
    )
```

Here, `insertion_is_valid` stands for a task-specific instantaneous predicate.
It returns one Boolean per parallel environment and includes all required
alignment, depth, and motion checks. This example does not add a temporal
requirement.

Predicates are ordinary callables invoked with `env`. The tracker does not
perform Isaac Lab manager preparation on them or their bound arguments.
Predicates that need resolved scene indices must arrange that preparation
explicitly; manager-term classes are not automatically constructed or reset.

An objective supplies exactly one of `sequence`, `predicate_groups`, or
`children`. `sequence=[settled, lifted, placed]` is one ordered chain.
`predicate_groups={"can": [can_lifted, can_placed], "bottle": [bottle_placed]}`
describes independent named chains. Lists are explicit even for a single
predicate. Both forms accept `(predicate, score)` pairs when stages need
different weights. Only independent groups use `logical` and `K` to select
how many chains must complete.

## Evaluation and reset ownership

```text
TerminationManager.compute()
  -> TaskSuccessFromProgress
    -> ProgressTracker.step()
    -> ProgressTracker.is_complete()

ProgressTrackingRecorder.record_post_step()
  -> read progress state and events
  -> publish env.extras["progress_tracking"]

Environment resets selected environments
  -> record the finishing episode, including its final progress
  -> TerminationManager.reset(env_ids)
    -> TaskSuccessFromProgress.reset(env_ids)
      -> reset progress for those environments
      -> reset their recorded initial object rest poses
```

The single root manager term owns the tracker; individual predicates do not
need their own `ManagerTermBase` adapters. The recorder publishes existing
results. It neither evaluates predicates nor
advances progress. Reading the termination manager's cached result is passive
as well. Success and reset therefore work without the progress reporter.
No separate progress reset event is needed.

Predicates within a progress chain run in order. Each chain advances by at most
one position per environment step. Completed positions are remembered until
reset, so pick-and-place requires settling, then lifting, then placing.

Composite tasks combine child objectives under one root. Sequential tasks
activate their children in order, independently for each parallel environment.
The next child begins sampling on the following step. Composition retains
completed child milestones; requested final child states can additionally
require their final conditions to hold or not hold at completion.

## Follow-up: consecutive true samples

Consecutive-step predicates are not implemented by this change. Their
configuration can describe a predicate and a required sample count without
holding runtime state. A proposed spelling is:

```python
# Proposed interface, not an available API.
ForSteps(predicate=insertion_is_valid, steps=10)
```

The progress runner would bind a separate counter to each configured occurrence.
The counter would consume Boolean results and the occurrence's per-environment
activation mask. It would not query the scene or own a scheduling hook.

For each active environment, a true sample increments its counter up to the
threshold. A false sample clears the counter. Inactive environments do not
accumulate history; deactivation clears any streak. Reset clears only the
specified environments. Reads leave the counter unchanged.

The runner remains responsible for advancing each counter once per control
step and forwarding selective resets. A new chain position begins sampling on
the following step, preserving the existing stage-advancement rule. An
ordinary predicate remains an ordinary callable and needs no new base class.

The counter reports a current condition: a false sample clears it even after
the threshold was reached. The progress runner separately remembers completed
milestones. These are different semantics and should remain explicit.

The duration applies to the complete condition that must persist. For three
gears that must remain seated and still together for ten steps, apply one
counter to the conjunction of every gear's geometry and motion checks.
Counting only low velocity would permit an object to rest outside its target
for nine steps and succeed on its first seated step.

One step means one completed environment/control step. Ten samples depend on
the configured control rate and do not establish what happened between
samples. Nested temporal expressions, if supported, need a documented sampling
order: `ForSteps(ForSteps(condition, 3), 2)` reaches its threshold after four
consecutive true samples, not six.

## Validation

The implemented lifecycle needs coverage for:

- Success on the same step as the final required predicate.
- One chain advancement per environment step, with passive repeated reads.
- Final progress available to the episode recorder before reset.
- Selective reset of progress, event history, and recorded rest poses.
- Success evaluation without the progress reporter.
- Composite and sequential completion through the same progress tracker.
- Builder-owned success and an explicit empty-objective task.

The counter follow-up additionally needs coverage for threshold boundaries,
interrupted streaks, independent parallel environments, stage activation,
nested temporal expressions if supported, and reset forwarding through
composed objectives.
