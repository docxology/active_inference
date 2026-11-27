# Results

The course search returned the posterior lesson. The ordered learning path
passed. Reversing the order produced two unsatisfied prerequisites. The
separate in-memory graph held three concept nodes and two edges, and traversal
returned the prior, observation, and posterior concepts in sequence. A missing
destination returned an empty lookup result represented as `None`.

The categorical fixture produced eight belief vectors of length three. All
belief vectors were normalized, and the observations, actions, beliefs and
diagnostics matched across the repeated runs. The largest posterior component
is reported as confidence in the source field `accuracy`. The replay contains
no external outcome labels. The figure shows the actual posterior vectors.

![A synthetic prerequisite graph and seeded belief replay. The left panel shows the lesson order; the right panel shows the state probabilities from the recorded simulation.](../../output/figures/teaching_fixture.png){#fig:teaching-fixture width=95%}
