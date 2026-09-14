# MCTSEvaluator: A Detailed Analysis

An analysis of `com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator`
— the Monte Carlo Tree Search engine that drives the computer opponent in
Ultimate Tic Tac Toe. It covers the search algorithm as implemented, the
single- and multi-threaded execution models, and how the trained neural
networks are substituted into the classical algorithm.

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
responsible for making sure "player 1" means the right thing. The class is
deliberately game-agnostic where it can be, working through the `GameState` /
`GameAction` interfaces in the `gameindependant` package, with
Ultimate-Tic-Tac-Toe specifics entering only on the neural-network paths.

### Configuration surface

`EvaluatorConfiguration` is a 9-field record, so the same engine covers a wide
range of behaviour without subclassing. The four presets in `Resources` are:

| Preset  | C | rollout depth | compute time | threads | force play | max CPU | networks |
|---------|---|---------------|--------------|---------|------------|---------|----------|
| DEFAULT | 2 | 1000          | 10 s         | 5       | yes        | no      | none     |
| EASY    | 2 | 1000          | 10 s         | 1       | yes        | no      | none     |
| MEDIUM  | 2 | 1000          | 30 s         | 25      | yes        | no      | **value + policy** |
| HARD    | 2 | 1000          | 60 s         | 25      | no         | **yes** | none     |

Two things fall out of this table immediately:

- **EASY is the only preset that runs the single-threaded code path**, because
  it is the only one with `threads <= 1`.
- **HARD and MEDIUM are different algorithms, not just different budgets.**
  HARD is classical MCTS given 60 seconds and an unbounded thread pool. MEDIUM
  is the network-guided configuration. The difficulty ladder spans
  "less compute → more compute → a different search strategy".

---

## 2. Data structures

### The tree

`GenericTree<NodeData>` / `GenericTreeNode<NodeData>` — a parent/children
object graph (BSD-licensed code from Vivin Suresh Paliath). Each node holds a
`NodeData` payload carrying the two MCTS statistics:

```java
protected int numVisits;
protected double totalScore;
```

`GenericTreeNode` synchronizes its structural accessors and mutators
(`getChildren`, `setChildren`, `addChild`, `getChildAt`) on the node monitor,
so the tree itself is safe to grow from several threads at once.

### Two node payloads, one important trade-off

| Class | Stores | `getGameState()` |
|-------|--------|------------------|
| `NodeData` | the `GameState` object itself | returns the stored reference |
| `BitSetBasedUTTTNodeData` | a `BitSet` encoding of the board | **constructs a new `BoardState` on every call** |

The root uses `NodeData` (line 67). Every expanded child uses
`BitSetBasedUTTTNodeData` (line 426). This is a deliberate memory-for-CPU
trade, and it is what makes deep search possible at all: a full `BoardState`
holds a `SubBoardState[3][3]` object graph, so storing millions of them would
exhaust the heap, while the `BitSet` form is a few dozen bytes. The price is
that reading a child's state re-decodes the board, which puts
`BoardState(BitSet)` on the hot path.

This is the single most consequential design decision in the class — it is the
difference between a tree that fits in memory at millions of nodes and one
that does not.

---

## 3. One iteration, phase by phase

`preformIteration(GenericTreeNode<NodeData> root)` (lines 200-255) is the whole
algorithm. Both the single- and multi-threaded drivers call exactly this
method; an iteration does not know which mode it is running under. That is
what keeps the two execution models from diverging.

### Phase 1 — Selection (lines 204-207)

```java
while (currentNode.hasChildren()) {
    currentNode = getBestChild(currentNode);
    depth++;
}
```

Descend until reaching a node with no children. The loop condition is
*structural* (`hasChildren()`) rather than the usual "fully expanded" test,
which works because this tree expands a node's entire legal-move set at once
(§3.2) — so "has children" and "is expanded" are the same property here.

`getBestChild` is the **first of the two swap points** where the
network-guided path diverges from the classical one (§6).

The classical selection rule is `getBestChildViaUCB` → `getUCB1`
(lines 394-402):

```
UCB1(node) = totalScore/numVisits + C * sqrt(ln(parent.numVisits) / numVisits)
```

with `numVisits <= 0` returning `POSITIVE_INFINITY`, so unvisited children are
always tried first. The scan short-circuits on the first infinity (line 368) —
a small but real optimization, since early in a node's life most of its
children are unvisited and there is no better candidate to find.

`C` is the exploration/exploitation dial, exposed in the UI and defaulting to 2.

### Phase 2 — Expansion (lines 213-238)

Guarded by `synchronized (currentNode)`, with an explicit rationale in the
comments: by the time a thread has descended to a leaf, contention on *that*
node's monitor is unlikely, so the lock is cheap where it is taken.

The branch structure is:

- **`numVisits == 0`** → don't expand. Evaluate this node and back up. This is
  the standard "visit a leaf once before expanding it" rule, which stops the
  tree from ballooning breadth-first.
- **`numVisits > 0` and no children** → `addChildren(currentNode)`, then
  evaluate the first child.
- **no children after expansion** → terminal position, so use
  `getGameState().getEvaluation()` directly.

`addChildren` (lines 409-430) generates every legal action, stamps the mover's
marker on it, and calls `simulateAction` to build each child state. It returns
early in two cases: the parent already has children, or the parent is terminal
(`getEvaluation() != 0`). The second check matters because `getActions()`
returns moves whenever empty squares exist, whether or not the game is already
decided — so without it the tree would grow past finished games.

### Phase 3 — Simulation / evaluation (lines 442-492)

`evaluateEndPoint` is the **second swap point**. Classically, `preformRollout`
plays uniformly random legal moves until the game resolves or
`maxRolloutDepth` (1000) is reached, then returns `getEvaluation()`.

Marker handling here is subtle and worth noting: `getActions()` always returns
actions marked for player 1, so the rollout calls `invertMarker()` when it is
player 2's turn (line 484). Getting this wrong produces a bot that plays both
sides as itself.

### Phase 4 — Backpropagation (lines 241-254)

The leaf's own statistics are updated inside the synchronized block. The walk
back up to the root is then done **lock-free**, a deliberate choice the source
explains:

```java
// this doesn't need to be synchronized I think even if another thread modifies
// one of the nodes before an iteration of this loop finishes, we shouldn't care
// becuase we are just adding to totals.
```

The reasoning is sound for this workload: MCTS statistics are already noisy
Monte Carlo estimates, visits and scores drift together so the score/visit
ratio stays representative, and the alternative — locking every ancestor up to
the root on every iteration — would serialize the threads precisely where they
overlap most. It trades a little statistical precision for throughput, which
is the right direction for a search that converges on volume.

Each step up negates the score (`scoreToAdd = -scoreToAdd`), so every node
stores its value from the perspective of the player to move at that node. This
is what lets the same UCB1 comparison work at every level without tracking
whose turn it is.

### Answer extraction

`getCurrentBestMove` (lines 500-512) returns the action leading to the
**most-visited** root child — the "robust child" criterion, rather than the
highest-scoring one. Under UCB1 the two converge, and visit count is the more
stable statistic of the pair, so this is the conventional and correct choice.

---

## 4. Single-threaded mode

Selected when `!useMaxCPU && THREADS <= 1` (line 88), which among the presets
means EASY. The driver is a plain loop (lines 165-174):

```java
while (searching) {
    preformIteration(tree.getRoot());
    iterations += 1;
    if (elapsed >= computeTime * 1000L)                     searching = false;
    else if (iterations >= EVALUATION_ITERATION_HARD_LIMIT)  searching = false;
}
```

Properties worth naming:

- **Exact iteration count.** `iterations` is a plain local — no estimation
  needed, which makes this path the reference for validating the parallel one.
- **Time granularity of one iteration.** The clock is checked after every
  iteration, so the search can overshoot its budget by at most one playout.
- **Both end conditions are live**: elapsed time, or the 50,000,000-iteration
  hard limit from `Resources`.

This is the version to reason against when judging the parallel implementation,
and keeping it as a separate, obviously-correct path is a useful debugging
affordance.

---

## 5. Multithreaded mode

### 5.1 Pool selection

```java
executor = useMaxCPU ? Executors.newCachedThreadPool()
                     : Executors.newFixedThreadPool(THREADS);
```

`useMaxCPU` (HARD) gives an unbounded cached pool that will grow to whatever
the machine will bear; otherwise a fixed pool of `THREADS`. MEDIUM and HARD
both request 25 threads. For the network-guided MEDIUM preset the threads
spend a share of their time inside ND4J rather than in pure tree work, so they
block less uniformly than a CPU-bound rollout workload would.

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

Seeded at `0 - 0 + 25`, this submits 25 tasks up front, then submits exactly
one replacement per completed task — a self-balancing window of roughly 25
outstanding tasks that is **independent of pool size**. The feeder
`Thread.sleep(1)`s between checks so it does not spin a core just to hand out
work.

The batching is what makes this cheap: executor submission overhead is
amortized across 100,000 iterations, while the per-iteration interrupt check
keeps shutdown responsive. Both ends of that trade were considered — a coarse
batch would normally mean a sluggish stop, and it does not here.

**Every task explores from `tree.getRoot()`.** That makes this *tree
parallelization* over one shared tree, rather than root parallelization
(independent trees merged at the end). Every thread's work therefore
contributes to the same statistics, so the search benefits from all of the
compute rather than averaging several weaker independent searches.

Termination is by wall clock (lines 114-119): on expiry, `searching = false`
and `executor.shutdownNow()` interrupts the in-flight batches.

### 5.3 The synchronization strategy

Three tiers of synchronization coexist, each chosen for its position in the
call graph:

1. **Per-node monitors** for structural work — the expansion block (line 213)
   and the UCB child scan (line 362), locked on the node being examined. This
   is the fine-grained tier, and contention on it falls off with depth because
   threads fan out as they descend.
2. **Lock-free backpropagation** (lines 248-254) — see §3.4 for the reasoning.
3. **Evaluator-level monitors** for aggregate telemetry —
   `setMaxDepth`, `recordTraversalStats`, `incrementPositionsSearched`,
   `incrementChildCreationRaceConditions`, `incrementCompletedTask` and the
   `getCompletedTasks` reader are `synchronized` instance methods, since they
   maintain counters that belong to the search as a whole rather than to any
   node.

The interesting property is that the tiers are ordered by how deep in the hot
path they sit: the deeper and more frequent the operation, the finer-grained
(or entirely absent) the locking.

### 5.4 Race instrumentation

The class counts its own contention. `addChildren` increments
`childCreationRaceConditions` when it finds a parent already expanded
(lines 412-415), and `preformSearch` reports it:

```
Detected: N node expansion race conditions
```

This is a good instinct for concurrent code. Two threads racing to expand the
same node is benign — one simply discards its work — but *how often* it happens
is exactly the kind of thing that is invisible until you measure it. Exposing
it as a number makes the shared-tree contention observable while tuning thread
counts, instead of something to be assumed about.

### 5.5 Reporting

Because work is batched, the multithreaded path reports an estimate rather
than an exact count:

```java
"Determined best move using roughly: " + getCompletedTasks() * BATCH + " iterations"
```

Only completed batches are counted, so the figure is a conservative lower
bound on the work actually done — the honest way to report it, and the reason
the word "roughly" is in the output. The search itself is bounded by the clock.

Alongside it, `preformSearch` reports maximum depth reached, states searched,
average traversal time, and the race count — a small built-in profiler that
makes the effect of a configuration change visible immediately.

---

## 6. Network-guided vs. classical search

The neural networks do not sit beside the search — they are **substituted into
two specific phases of it**. Both substitutions are independently optional and
both fall back to the classical behaviour, which is what lets a single class
serve both modes.

```
                  CLASSICAL                        NETWORK-GUIDED
Selection    UCB1 over child statistics   ←→   sample from policy-net PMF
Evaluation   uniform-random rollout       ←→   value-net forward pass
```

This is the architectural idea worth taking away from the class: MCTS is a
framework with replaceable parts, and "adding AI" means swapping the policy
and evaluation functions, not bolting on a separate system.

### Swap point 1 — Selection

`getBestChild` (lines 268-286): with a policy network present, encode the
state, run a forward pass, mask illegal moves to zero, renormalize into a
probability mass function, and **sample** a move from it
(`getChildToExploreFromPolicyNetworkOutput`, lines 297-326). The sampled action
is then matched against the node's children by value equality and that child
is descended into.

Note this samples rather than taking the `argmax`: the PMF is accumulated into
a CDF and drawn against a uniform random via binary search (lines 328-348).
Randomized selection preserves exploration where `argmax` would be greedy —
which matters because the network's probabilities have replaced UCB1's
confidence bound as the only source of exploration pressure.

The masking step is the essential correctness detail. A network trained on
board tensors offers no hard guarantee of proposing a legal move, so all 81
cells are tested with `currentState.isLegal(...)` and zeroed if illegal before
renormalizing, with a second check after sampling that falls back to the first
legal action. This is the kind of detail that is easy to skip and expensive to
debug later.

### Swap point 2 — Evaluation

`evaluateEndPoint` (lines 442-461): with a value network present, encode the
state, take `output(...).getDouble(0, 0)`, and **negate it when it is player
2's turn** — the network always sees the board from player 1's perspective, so
the sign is corrected to the caller's convention. Any failure is recorded in
`warnings` and degrades to a random rollout, so a network problem costs search
quality rather than crashing the game.

Theoretically this is the substitution with the clearest payoff: it replaces a
high-variance estimate (a single random playout, which for Ultimate Tic Tac Toe
can be a nearly meaningless sequence of moves) with a low-variance learned one
— the same substitution AlphaGo made.

### The shared encoder

Both networks consume the same tensor; `getPolicyNetworkInputV1` is literally
`return getValueNetworkInputV1(state);` (machinelearning/`Resources.java:258`).
The encoding is a 4-channel one-hot over the 9 sub-boards:

```
shape [1, 9, 3, 3, 4]   (batch, major index, minor row, minor col, channel)
  channel 0: X present
  channel 1: O present
  channel 2: empty
  channel 3: sub-board active  (broadcast across all 9 cells of that board)
```

Channel 3 is what makes the encoding sufficient rather than lossy. In this game
legality depends on which sub-board is active, so a bare 81-cell occupancy grid
would be an incomplete description of the position — two identical-looking
boards can have completely different legal move sets. Channels 0-2 are one-hot
rather than a single signed value, which is the friendlier representation for a
convolution to learn from.

### Cost profile

The two swaps have opposite performance characteristics, and understanding that
asymmetry explains the whole difficulty ladder:

- **Value network (evaluation)** — one forward pass **per iteration**,
  replacing a rollout that may itself be hundreds of `simulateAction` calls.
  This can be a net win on cost as well as on variance.
- **Policy network (selection)** — one forward pass **per node per level of
  descent**. A ten-ply descent means ten forward passes before a single
  position is evaluated, each with its own board encoding, legality mask and
  renormalization. Iterations per second drops sharply against UCB1's
  arithmetic-only comparison.

That asymmetry is why network-guided MCTS is judged on **quality per second**
rather than iterations per second, and why MEDIUM's 30-second budget buys far
fewer iterations than HARD's 60 while still playing well: each iteration is
better directed and better evaluated.

---

## 7. What this design demonstrates

- **Phase substitution as an architecture.** Classical and network-guided MCTS
  coexist in one implementation, each network is independently optional, and
  both degrade to the classical path when absent or failing. That is why the
  difficulty ladder can span two genuinely different search strategies without
  a second engine.
- **The task-deficit feeder** (§5.2) — a pool-size-independent way to keep
  workers saturated without unbounded queue growth or per-iteration submission
  overhead.
- **Batched iterations with per-iteration interrupt checks** — executor
  overhead amortized 100,000× while keeping shutdown immediate.
- **Synchronization graded by hot-path depth** (§5.3) — fine-grained node locks
  where threads collide structurally, lock-free accumulation where precision is
  cheap to trade for throughput, and instance-level locks only for
  search-wide counters.
- **Self-instrumentation** — depth, states searched, traversal timing and
  expansion races are all reported, turning a concurrent search into something
  observable rather than a black box.
- **A memory-conscious node representation** (§2) — the `BitSet` payload is
  what allows a tree of millions of nodes.
- **The unglamorous correctness details done right** — legality masking on the
  policy output, perspective negation on the value output, and marker inversion
  in rollouts. These are the three things that quietly ruin a game AI and are
  hardest to notice from the outside.

---

## 8. Image generation prompt

A prompt for generating a portfolio diagram of this class, written to be pasted
into an image model (Midjourney, DALL·E, Imagen, Ideogram, etc.).

> **Note on text in generated images.** Image models reliably mangle this
> quantity of small text. Treat the output as the *visual* — composition,
> palette, flow — and expect to redraw the labels in Figma, Excalidraw, or SVG
> afterwards. If you want correct labels first time, Mermaid or Graphviz is the
> better tool; use this prompt for the hero image that makes someone stop
> scrolling.

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
 - upper half "CLASSICAL": a UCB1 formula block and a dice icon, drawn in
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
a self-balancing work queue. Two or three tiny padlock glyphs sit on individual
tree nodes, indicating fine-grained per-node locking.

A 3x3 grid of 3x3 Ultimate Tic Tac Toe boards appears as a subtle, very
low-contrast watermark behind the tree, tying the diagram to the game without
competing with the foreground.

Palette: near-white or very light warm grey background (#FAFAF8), deep
charcoal linework (#2A2A2E), cool slate blue for the classical elements
(#3B6EA5), warm amber for the neural-network elements (#D98A2B), one single
vivid accent for the highlighted selection path (#E4572E). Thin consistent 2px
stroke weights. Muted, professional, high contrast between the blue and amber
families so the two modes read as distinct at a glance. Typography: small clean
geometric sans-serif labels in sentence case, sparse, only where called out
above. Crisp, legible, uncluttered, suitable for a portfolio case-study header.
```

### Why the prompt is built this way

- **Four panels, one per axis of the analysis** — the iteration cycle (§3), the
  two modes (§6), and the parallel execution model (§5) — so a viewer takes in
  the structure of the whole class at a glance.
- **The dashed arrows landing on stages 1 and 3** carry the single most
  important idea: the networks are *substituted into* two phases of an
  otherwise unchanged algorithm. Most MCTS+NN diagrams draw the network as a
  separate box beside the search, which gets the architecture wrong.
- **Opacity-as-visit-count** shows the asymmetric tree growth that is the whole
  point of MCTS, without requiring a single number to be legible.
- **The blue/amber split** gives the classical-vs-guided contrast a colour
  language that survives being scaled down to a thumbnail.
- **The worker lanes all feeding one tree**, plus the per-node padlocks, convey
  tree parallelism over shared state — the most technically interesting
  property of the implementation and the best thing to be asked about in an
  interview.
