# Knowledge repositories and courses

`KnowledgeRepository` loads JSON lesson definitions from directory roots
configured through `KnowledgeRepositoryConfig`. Indexed fields include
content type, difficulty and tags. Its text search checks titles and
descriptions with case-insensitive substring matching.

A lesson has an ID, title, content type, difficulty, description, prerequisites,
tags, learning objectives, content and metadata. The JSON loader admits
foundation, mathematics, implementation and application definitions. Put
ordered course definitions in `metadata/learning_paths.json`. Each course
lists the lesson IDs in delivery order.

The [offline example](QUICK_START.md) writes a temporary course with three
lessons on prior beliefs, observations and posterior updates. It records
`validate_learning_path` for the original and reversed orders and exports
the posterior lesson's prerequisite graph. The reversed sequence produces
two unsatisfied prerequisites.

`KnowledgeGraphManager` is a separate in-memory service. The example explicitly
adds the same lessons and prerequisite edges, then calls `find_shortest_path`.
That method counts edges and traverses them in either direction. Use
`KnowledgeRepository.validate_learning_path` for course order.

The checked-in concept and domain material can be browsed in the
[knowledge tree](https://github.com/docxology/active_inference/tree/main/knowledge).
The fixture supplies its own course JSON to make the replay self-contained.
