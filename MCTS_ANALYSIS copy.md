# MCTSEvaluator: A Detailed Analysis

An analysis of `com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator`
(641 lines) — the Monte Carlo Tree Search engine that drives the computer
opponent in Ultimate Tic Tac Toe.

Line references are to `MCTSEvaluator.java` unless noted.

---

## 1. What the class is

`MCTSEvaluator` is constructed with one `GameState` and one
`EvaluatorConfiguration`, and answers a single question: **what is the best
move for the player holding marker `1`?**

```java
MCTSEvaluator evaluator = new MCTSEvaluator(stateToPass, config);
GameAction move = evaluator.preformSearch();
```

The evaluator is single-use and stateful — it owns a search tree that it grows
in place, then reads the answer off the root's children. The caller is
responsible for making sure "player 1" means the right thing; the class itself
is deliberately game-agnostic where it can be, working through the
`GameState` / `GameAction` interfaces (`gameindependant` package), with
Ultimate-Tic-Tac-Toe specifics leaking in only on the neural-network paths.

### Configuration surface

`EvaluatorConfiguration` is a 9-field record. The four presets in
`Resources` are:

| Preset  | C | rollout depth | compute time | threads | stupidity | force play | max CPU | networks |
|---------|---|---------------|--------------|---------|-----------|------------|---------|----------|
| DEFAULT | 2 | 1000          | 10 s         | 5       | 0         | yes        | no      | none     |
| EASY    | 2 | 1000          | 10 s         | 1       | 100       | yes        | no      | none     |
| MEDIUM  | 2 | 1000          | 30 s         | 25      | 50        | yes        | no      | **value + policy** |
| HARD    | 2 | 1000          | 60 s         | 25      | 25        | no         | **yes** | none     |

Two things fall out of this table immediately:

- **EASY is the only preset that runs the single-threaded code path**, because
  it is the only one with `threads <= 1`.
- **HARD is not the "most AI" setting — it is the least.** HARD is classical
  MCTS given 60 seconds and an unbounded thread pool. MEDIUM is the
  network-guided configuration. The difficulty ladder is really
  "less compute → more compute → different algorithm".

---

## 2. Data structures

### The tree

`GenericTree<NodeData>` / `GenericTreeNode<NodeData>` — a plain
parent/children object graph (BSD-licensed code from Vivin Suresh Paliath).
Each node holds a `NodeData` payload with the two MCTS statistics:

```java
protected int numVisits;
protected double totalScore;
```

`GenericTreeNode` synchronizes its structural mutators (`setChildren`,
`addChild`, `getChildren`, `getChildAt`) on the node monitor. `getParent()` is
not synchronized, but `parent` is only ever written during `setChildren`,
before a node becomes reachable from the traversal.

### Two node payloads, one important trade-off

| Class | Stores | `getGameState()` |
|-------|--------|------------------|
| `NodeData` | the `GameState` object itself | returns the stored reference |
| `BitSetBasedUTTTNodeData` | a `BitSet` encoding of the board | **constructs a new `BoardState` on every call** |

The root uses `NodeData` (line 67). Every expanded child uses
`BitSetBasedUTTTNodeData` (line 426). This is a deliberate
memory-for-CPU trade: a full `BoardState` holds a `SubBoardState[3][3]` object
graph, so storing millions of them would exhaust the heap — the `BitSet` form
is a few dozen bytes. The cost is that **every read of a child's state
re-decodes the whole board**, and `getGameState()` sits in the hottest path in
the class.

---

## 3. One iteration, phase by phase

`preformIteration(GenericTreeNode<NodeData> root)` (lines 200-255) is the whole
algorithm. Both the single- and multi-threaded drivers call exactly this
method; nothing about an iteration knows which mode it is in.

### Phase 1 — Selection (lines 204-207)

```java
while (currentNode.hasChildren()) {
    currentNode = getBestChild(currentNode);
    depth++;
}
```

Descend until a node with no children is reached. Note the loop condition is
*structural* (`hasChildren()`), not the usual "fully expanded" test — this tree
expands a node's entire legal-move set at once (§3.2), so "has children" and
"is expanded" are the same thing here.

`getBestChild` is the **first of the two swap points** where the AI path
diverges from the traditional one (§6).

The classical selection rule is `getBestChildViaUCB` → `getUCB1` (lines
394-402):

```
UCB1(node) = totalScore/numVisits + C * sqrt(ln(parent.numVisits) / numVisits)
```

with `numVisits <= 0` returning `POSITIVE_INFINITY` so unvisited children are
always taken first. The scan short-circuits on the first infinity (line 368) —
a small but real win, since early in a node's life most children are unvisited.
`C` defaults to 2 across all four presets; the exploration/exploitation dial is
exposed in the UI but never actually varied by the presets.

### Phase 2 — Expansion (lines 213-238)

Guarded by `synchronized (currentNode)`, with an explicit rationale in the
comments: by the time a thread reaches a leaf it is deep in the tree, so
contention on *that* node's monitor is unlikely.

The branch structure is:

- **`numVisits == 0`** → don't expand. Evaluate this node and back up. (The
  standard "visit a leaf once before expanding it" rule.)
- **`numVisits > 0` and no children** → `addChildren(currentNode)`, then
  evaluate `getChildAt(0)`.
- **no children after expansion** → terminal state, use
  `getGameState().getEvaluation()` directly.

`addChildren` (lines 409-430) generates every legal action, stamps the mover's
marker on it, and calls `simulateAction` to build each child state. It bails
early in two cases: the parent already has children (counted as a race, §5.4),
or the parent is terminal (`getEvaluation() != 0`) — the latter matters because
`getActions()` returns moves whenever empty squares exist, terminal or not.

> **Known bias:** the newly expanded node always rolls out **child 0**
> (line 230). The source comment already flags this: the move ordering returned
> by `getActions()` systematically biases which region of the board gets
> explored first. A random child would remove it.

### Phase 3 — Simulation / evaluation (lines 442-492)

`evaluateEndPoint` is the **second swap point**. Traditionally:

`preformRollout` plays uniformly random legal moves until the game resolves or
`maxRolloutDepth` (1000) is hit, then returns `getEvaluation()`. The depth cap
is a safety rail against a non-terminating playout. Marker handling is subtle:
`getActions()` always returns actions marked for player 1, so the rollout calls
`invertMarker()` when it is player 2's turn (line 484).

### Phase 4 — Backpropagation (lines 241-254)

The leaf is updated inside the synchronized block; the walk to the root is
**deliberately left unsynchronized**, with this reasoning in the source:

```java
// this doesn't need to be synchronized I think even if another thread modifies
// one of the nodes before an iteration of this loop finishes, we shouldn't care
// becuase we are just adding to totals.
```

Each step up negates the score (`scoreToAdd = -scoreToAdd`), so a value is
stored from the perspective of the player to move at that node. See §5.3 for
why "just adding to totals" is not actually safe.

### Answer extraction

`getCurrentBestMove` (lines 500-512) returns the action leading to the
**most-visited** root child — the "robust child" criterion, not the
highest-scoring one. Under UCB1 these converge, and visit count is the more
stable statistic, so this is the conventional and correct choice.

---

## 4. Single-threaded mode

Selected when `!useMaxCPU && THREADS <= 1` (line 88), which among the presets
means EASY only. The entire driver is a `while` loop (lines 165-174):

```java
while (searching) {
    preformIteration(tree.getRoot());
    iterations += 1;
    if (elapsed >= computeTime * 1000L)                  searching = false;
    else if (iterations >= EVALUATION_ITERATION_HARD_LIMIT) searching = false;
}
```

Properties worth naming:

- **Exact iteration count.** `iterations` is a plain local — no estimation.
- **Time granularity is one iteration.** The clock is checked after every
  iteration, so the search overshoots its budget by at most one playout.
- **Both end conditions are live**: elapsed time *or* the 50,000,000-iteration
  hard limit from `Resources`.
- **No synchronization cost, but the locks are still taken.** The same
  `synchronized` blocks and `synchronized` counter methods execute; they are
  uncontended, so the JVM biases/elides them cheaply, but the code is not
  specialized for the single-threaded case.

This path is the reference implementation: simple, deterministic in structure,
and the one to reason against when judging the parallel version.

---

## 5. Multithreaded mode

### 5.1 Pool selection

```java
executor = useMaxCPU ? Executors.newCachedThreadPool()
                     : Executors.newFixedThreadPool(THREADS);
```

`useMaxCPU` (HARD) gives an unbounded cached pool; otherwise a fixed pool of
`THREADS`. Note MEDIUM and HARD both request **25 threads**, well above the
core count of a typical machine — for a CPU-bound workload that is
oversubscription, though for the network-guided MEDIUM preset the threads
spend time inside ND4J and block less uniformly.

### 5.2 The task-deficit feeder

This is the most interesting piece of engineering in the class. Rather than
submitting one task per iteration, work is batched:
`PreformBatchOfIterationsTask` (lines 619-640) runs
`Resources.MULTTHREADED_BATCH_SIZE` = **100,000 iterations** per task, checking
`Thread.interrupted()` after each one.

The control loop (lines 112-138) keeps the queue at a steady depth:

```java
long taskDeficit = getCompletedTasks() - submittedTasks + 25;
for (int i = 0; i < taskDeficit; i++) {
    submittedTasks++;
    executor.execute(new PreformBatchOfIterationsTask(BATCH, tree.getRoot()));
}
```

Seeded at `0 - 0 + 25`, this submits 25 tasks, then submits exactly one
replacement per completion — a self-balancing window of ~25 outstanding tasks,
independent of pool size. The loop `Thread.sleep(1)`s between checks so the
feeder thread does not spin.

**Every task explores from `tree.getRoot()`.** That makes this *tree
parallelization* over one shared tree — not root parallelization (independent
trees, merged at the end). The docstring on `preformIteration` mentions
exploring initial children separately, but the implemented behaviour is a
single shared tree.

Termination is by wall clock (lines 114-119): on expiry, `searching = false`
and `executor.shutdownNow()` interrupts in-flight batches. Because a batch is
100,000 iterations but interruption is checked every iteration, the coarse
batch size costs nothing in responsiveness.

### 5.3 The locking strategy, and where it leaks

Three tiers of synchronization coexist:

1. **Per-node monitors** — the expansion block (line 213) and the UCB child
   scan (line 362), locked on the node being examined. Fine-grained, and
   contention falls off with depth.
2. **Unsynchronized backprop** — lines 248-254, by design.
3. **Evaluator-instance monitors** — the statistics methods
   (`setMaxDepth`, `recordTraversalStats`, `incrementPositionsSearched`,
   `incrementChildCreationRaceConditions`, `incrementCompletedTask`,
   and the `getCompletedTasks` reader) are `synchronized` instance methods,
   i.e. all contending on the *same* monitor.

Tier 3 is the scalability ceiling. `setMaxDepth` and `recordTraversalStats` are
called **once per iteration each** (lines 208-209), so every iteration on every
thread acquires the single evaluator lock twice — purely to maintain telemetry.
At 25 threads this is a global serialization point in the hottest loop; the
diagnostics measuring the search are themselves throttling it. `LongAdder`,
`AtomicLong`, or an `AtomicInteger` CAS loop for the max would remove it
entirely.

Tier 2 is a correctness question rather than a performance one. `numVisits` and
`totalScore` are ordinary non-`volatile` fields mutated by
`numVisits += 1` / `totalScore += delta` — read-modify-write, so concurrent
backprop through a shared ancestor **loses updates**. Near the root, where
every iteration passes through, contention is maximal. The practical impact is
mild and self-limiting: MCTS statistics are already noisy estimates, a lost
visit is a rounding error against millions of iterations, and both counters
drift *together*, so the score/visit ratio stays roughly right. But it also
means the reported totals are a lower bound, and there is no memory-visibility
guarantee that one thread's updates are ever seen by another. The fix
(`AtomicInteger` + `DoubleAdder`, or locking the node) is cheap at the root and
would matter most exactly where the bias is worst.

Also shared and contended: `public static Random rand` (line 45), the source of
every rollout move for every thread. `Random` is thread-safe but does so with a
CAS loop on a single seed — under 25 threads pulling from it in a tight loop
this is measurable contention with no upside. `ThreadLocalRandom` is the
drop-in answer.

### 5.4 Race instrumentation

The class counts its own races. `addChildren` increments
`childCreationRaceConditions` when it finds a parent already expanded
(lines 412-415), and `preformSearch` reports it:

```
Detected: N node expansion race conditions
```

This is a genuinely good instinct — the benign-but-wasteful case (two threads
racing to expand the same node, one discarding its work) is made *visible*
rather than assumed away. It turns an invisible concurrency property into a
number you can watch while tuning.

### 5.5 Reporting differences

The multithreaded path cannot count iterations exactly, and says so:

```java
"Determined best move using roughly: " + getCompletedTasks() * BATCH + " iterations"
```

Only *completed* batches are counted, so the figure undercounts by up to
25 × 100,000 in-flight iterations, and — since a time-triggered
`shutdownNow()` almost always lands mid-batch — it systematically undercounts.
It also means the multithreaded path has **no iteration cap at all**: only the
clock stops it, whereas the single-threaded path honours
`EVALUATION_ITERATION_HARD_LIMIT`.

---

## 6. AI-guided vs. traditional

The neural networks do not sit beside the search — they are **substituted into
two specific phases of it**. Both substitutions are optional and both fall back
to the classical behaviour, which is what lets one class serve both modes.

```
                  TRADITIONAL                      AI-GUIDED
Selection    UCB1 over child statistics   ←→   sample from policy-net PMF
Evaluation   uniform-random rollout       ←→   value-net forward pass
```

### Swap point 1 — Selection

`getBestChild` (lines 268-286): if a policy network is present, encode the
state, run a forward pass, mask illegal moves to zero, renormalize into a
probability mass function, and **sample** a move from it
(`getChildToExploreFromPolicyNetworkOutput`, lines 297-326).

Note this is sampling, not `argmax`: the PMF is turned into a CDF and drawn
against a uniform random (lines 328-348, with a binary search over the CDF).
Randomized selection preserves some exploration where `argmax` would be greedy
— a reasonable choice given that the network's probabilities, not UCB1's
confidence bound, are now the only exploration pressure.

The masking step is the essential correctness detail: a network trained on
board tensors has no hard guarantee of proposing a legal move, so every one of
the 81 cells is tested with `currentState.isLegal(...)` and zeroed if illegal
before renormalizing. There is a second belt-and-braces check after sampling
that falls back to the first legal action.

### Swap point 2 — Evaluation

`evaluateEndPoint` (lines 442-461): if a value network is present, encode the
state, take `output(...).getDouble(0, 0)`, and **negate it when it is player
2's turn** — the network always sees the board from player 1's perspective, so
the sign has to be corrected to the caller's convention. Any exception is
caught, recorded in `warnings`, and degraded to a random rollout.

This is the substitution with the clearest theoretical payoff: it replaces a
high-variance estimate (one random playout) with a low-variance one (a learned
evaluation), which is the same move AlphaGo made. The cost is per-node latency
and a hard dependency on the network's quality.

### The shared encoder

Both networks consume the same tensor. `getPolicyNetworkInputV1` is literally
`return getValueNetworkInputV1(state);` (machinelearning/`Resources.java:258`).
The encoding is a 4-channel one-hot over the 9 sub-boards:

```
shape [1, 9, 3, 3, 4]   (batch, major index, minor row, minor col, channel)
  channel 0: X present
  channel 1: O present
  channel 2: empty
  channel 3: sub-board active  (broadcast across all 9 cells of the board)
```

Channel 3 is what makes the encoding sufficient: in this game legality depends
on which sub-board is active, so a bare 81-cell occupancy grid would be an
incomplete description of the position. Channels 0-2 are one-hot rather than a
single signed value, which is the friendlier representation for a convolution.

### Cost profile of the AI path

The two swaps have opposite performance characteristics, and this is the
central engineering tension in the class:

- **Value network (evaluation)** — one forward pass **per iteration**,
  replacing a rollout that could itself be hundreds of `simulateAction` calls.
  This can be a net *win* on cost as well as on variance.
- **Policy network (selection)** — one forward pass **per node per level of
  descent**. A 10-ply descent means 10 forward passes before a single
  evaluation happens, each preceded by a `BoardState` re-materialization from
  the `BitSet` (twice per call, lines 270-275), plus an 81-iteration legality
  mask and a renormalization. Iterations per second collapses relative to
  UCB1's arithmetic-only comparison.

That asymmetry is why a network-guided MCTS is usually judged on *quality per
second*, not iterations per second — and why MEDIUM's 30-second budget buys far
fewer iterations than HARD's 60.

Concurrency caveat: with MEDIUM's 25 threads, `MultiLayerNetwork.output()` is
called concurrently on one shared network instance. DL4J does not document
`output()` as thread-safe for a shared `MultiLayerNetwork`; the only protection
here is the `try/catch` in `evaluateEndPoint`, which converts a failure into a
silent downgrade to rollout plus a line in `warnings`. Per-thread network
copies, or a single inference-serving thread, would make this deterministic.

---

## 7. Findings

Verified against the source while writing this document. Ordered by impact.

### 7.1 The policy network's output is computed and then discarded

`getBestChild` matches the sampled action against the node's children with
**reference equality** (line 278):

```java
if (child.getData().getActionTaken() == actionToTake) {
```

`actionToTake` comes from `getActionFromFlatIndex`, which **returns a freshly
allocated** `UltimateTickTacToeGameAction`
(machinelearning/`Resources.java:58`). It is never the same object as a child's
stored action, so the comparison is always false, the loop always falls
through, and line 284 `return getBestChildViaUCB(currentNode);` always runs.

`GameAction` **does** override `equals` and `hashCode`
(`GameAction.java:58`, `:68`), so `.equals(...)` is available and would work.

Net effect: whenever a policy network is configured — i.e. the entire MEDIUM
preset — the engine pays for a forward pass, two `BoardState` reconstructions,
an 81-cell legality mask, a renormalization and a CDF sample **at every node of
every descent**, then throws the result away and selects by UCB1 anyway. The
policy network is pure overhead today. Changing `==` to `.equals(...)` is the
one-character-class fix that switches the feature on — and it should be
measured immediately after, because it will make MEDIUM dramatically slower per
iteration in exchange for better-directed search.

### 7.2 `stupidity` is inert

`STUPIDITY` is assigned from the config (line 71) and **never read anywhere in
the codebase** — the only two occurrences in `src/main/java` are its
declaration (line 37) and that assignment. It is surfaced in the UI as a
slider, persisted into saved evaluator JSON, and carried through the record,
but nothing consumes it. EASY (stupidity 100) and HARD (stupidity 25) differ
only in their other fields.

*This also means the `README.md` feature list is wrong: it describes stupidity
as "the chance the engine deliberately plays a non-optimal move". That was my
error in writing it — the intent is clear from the name and presets, but the
behaviour does not exist. It should be corrected or the feature implemented.*

### 7.3 Telemetry locks serialize the hot loop

`setMaxDepth` and `recordTraversalStats` are `synchronized` instance methods
called once per iteration each (lines 208-209), so every thread acquires the
same evaluator-wide monitor twice per iteration. At 25 threads this is likely
the dominant scaling limit, and it exists only to maintain diagnostics.
`LongAdder`/`AtomicLong` would eliminate it.

### 7.4 Lost updates in backpropagation

`numVisits`/`totalScore` are non-`volatile` fields updated with non-atomic
read-modify-write, from an intentionally unsynchronized loop (lines 248-254).
Concurrent passes through shared ancestors lose updates, worst at the root.
Impact is mild — counters drift together, so ratios survive, and MCTS tolerates
noise — but the reported totals are a lower bound with no visibility guarantee.

### 7.5 Shared `static Random`

`public static Random rand` (line 45) serves every rollout on every thread,
CAS-contending on one seed. `ThreadLocalRandom` is a free fix.

### 7.6 Concurrent inference on a shared DL4J network

See §6. Guarded only by a catch-and-degrade.

### 7.7 Smaller items

- **Expansion always rolls out child 0** (line 230) — move-ordering bias, already
  flagged in a source comment.
- **`getCurrentBestMove` uses `>=`** (line 504), so ties break toward the
  last-enumerated child rather than the first. Harmless, but arbitrary.
- **`enum EndCondition { TIME, ITERATIONS }`** (line 614) is dead code — zero
  references in `src`. The two conditions it names are hardcoded instead.
- **`Thread.interrupted()`** at line 120 *clears* the interrupt flag; it is the
  static draining call, not `isInterrupted()`. Harmless here (the loop breaks
  immediately) but easy to misread.
- **Iteration count is an undercount** in multithreaded mode (§5.5), and the
  multithreaded path has no iteration cap.
- **`BitSet` → `BoardState` re-materialization** (`BitSetBasedUTTTNodeData`)
  happens on every `getGameState()` call, including twice per policy-net
  selection. A correct memory trade, but the hot path calls it more than it
  needs to.

### 7.8 What the class gets right

Worth stating plainly, because the list above is a list of defects and the
design is better than that implies:

- The **phase-substitution architecture** is the right shape. Classical and
  network-guided MCTS coexist in one implementation, each network is
  independently optional, and both degrade to the classical path on absence or
  failure. That is why the difficulty ladder can span two different algorithms
  without a second engine.
- The **task-deficit feeder** (§5.2) is a genuinely elegant solution to keeping
  a pool saturated without unbounded queue growth or per-iteration submission
  overhead, and it is pool-size-independent.
- **Batching iterations** amortizes executor overhead by 100,000× while
  per-iteration interrupt checks keep shutdown responsive — both ends of that
  trade were thought about.
- **Instrumenting its own race conditions** (§5.4) is a level of concurrency
  self-awareness most hobby MCTS implementations do not have.
- **Legality masking** on the policy output, and the **perspective negation** on
  the value output, are exactly the two details that are easy to get wrong when
  wiring a network into a search and expensive to debug afterwards.
- The **`BitSet` node payload** is the difference between a tree that fits in
  memory at millions of nodes and one that does not.

---

## 8. Image generation prompt

A prompt for generating a portfolio diagram of this class. It is written to be
pasted into an image model (Midjourney, DALL·E, Imagen, Ideogram, etc.).

> **Note on text in generated images.** Image models reliably mangle
> quantities of small text like this. Treat the output as the *visual* —
> composition, palette, flow — and expect to redraw the labels in Figma,
> Excalidraw, or SVG afterwards. If you want correct labels first time, a
> Mermaid or Graphviz diagram is the better tool; use this prompt for the
> hero image that makes someone stop scrolling.

### The prompt

```
A clean, modern technical architecture diagram illustrating a Monte Carlo Tree
Search engine for the game Ultimate Tic Tac Toe, in the visual style of a
high-end software engineering blog or conference slide. Landscape 16:9, flat
vector illustration, generous negative space, no photorealism, no 3D bevels,
no drop shadows.

CENTER: a search tree growing downward over four levels. A single root node at
top, branching to 5 children, then to 9 grandchildren, with the lowest level
fading to thin translucent outlines to suggest an enormous unexplored frontier.
Nodes are rounded rectangles. Each node shows two small statistics as abstract
glyphs: a visit-count bar and a score value. Node fill opacity encodes visit
count, so a few heavily-visited nodes are richly saturated while most are pale
— making the asymmetric, deeply-explored principal variation visually obvious.
One path from root to leaf is highlighted as a thick glowing line: the current
selection path.

LEFT PANEL, labelled "ONE ITERATION": four numbered stages stacked vertically,
connected by a downward arrow, each with a small distinct icon —
1 SELECTION (a branching arrow choosing between two paths),
2 EXPANSION (a node splitting into several new children),
3 EVALUATION (a dice icon AND a small neural-network icon side by side,
  joined by a two-way toggle switch to show they are interchangeable),
4 BACKPROPAGATION (an upward arrow along the tree with alternating plus and
  minus signs, indicating the score negates at each level).

RIGHT PANEL, labelled "TWO MODES", split into two clearly contrasted halves:
 - upper half "TRADITIONAL": a UCB1 formula block and a dice icon, drawn in
   cool blue tones.
 - lower half "NEURAL-GUIDED": two stacked network icons labelled POLICY and
   VALUE, drawn in warm amber tones, with the policy icon emitting a small
   9x9 probability heatmap grid and the value icon emitting a single gauge.
 Two dashed amber arrows leave this panel and land precisely on stages 1 and 3
 of the left panel, visually asserting that the networks substitute INTO those
 two phases rather than replacing the search.

BOTTOM STRIP, labelled "PARALLEL EXECUTION": a horizontal row of 6 worker
lanes, each drawn as a rounded capsule containing a stack of small task
blocks, all feeding arrows upward into the single shared tree in the center to
convey one shared tree with many workers. A small queue of pending task blocks
sits to the left of the lanes with a circular replenishment arrow, representing
a self-balancing work queue. Two or three tiny lock glyphs sit on individual
tree nodes, indicating fine-grained per-node locking. One small warning glyph
marks where two workers touch the same node.

A 3x3 grid of 3x3 Ultimate-Tic-Toe boards appears as a subtle, very
low-contrast watermark behind the tree, tying the diagram to the game without
competing with the foreground.

Palette: near-white or very light warm grey background (#FAFAF8), deep
charcoal linework (#2A2A2E), cool slate blue for the classical/traditional
elements (#3B6EA5), warm amber for the neural-network elements (#D98A2B), one
single vivid accent for the highlighted selection path (#E4572E). Thin
consistent 2px stroke weights. Muted, professional, high contrast between the
blue and amber families so the two modes read as distinct at a glance.
Typography: small clean geometric sans-serif labels in sentence case, sparse,
only where called out above. Crisp, legible, uncluttered, suitable for a
portfolio case-study header.
```

### Why the prompt is built this way

- **Four panels, one per axis of the analysis** — the iteration cycle (§3), the
  two modes (§6), and the parallel execution model (§5) — so a viewer gets the
  structure of the whole class from one glance.
- **The dashed arrows landing on stages 1 and 3** are the single most important
  idea to convey: the networks are *substituted into* two phases of an
  otherwise unchanged algorithm. Most MCTS+NN diagrams draw the network as a
  separate box beside the search, which gets the architecture wrong.
- **Opacity-as-visit-count** shows the asymmetric tree growth that is the whole
  point of MCTS, without needing a single number to be legible.
- **The blue/amber split** gives the "traditional vs. AI" contrast a colour
  language that survives being scaled down to a thumbnail.
- **The lock glyphs and the warning glyph** hint at §5.3-5.4 — that shared-tree
  parallelism has real synchronization consequences — which is the part of this
  project worth talking about in an interview.
