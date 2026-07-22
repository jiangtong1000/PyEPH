# LLM context prompt

Fill in the placeholders and provide this prompt to an LLM that can read the tutorial
directory and your calculation files.

```text
Read README.md and the supplied tools in <TUTORIAL_DIR> before helping with my
customized Quantum ESPRESSO phonon workflow.

Use this technical model:

- QE supports GRID-style calculation and recollection of dynmat.Q.IRREP.xml.
- Merging partial-irrep dvscf files is an advanced, version-specific extension.
- dvscf records follow cumulative mode numbers, not irrep numbers.
- Workers must share identical displacement patterns and compatible QE/SCF provenance.
- File size alone cannot detect sparse, zero, missing, or incorrect records.
- Q-point, image, irrep, and mode counts are different quantities.
- QE build, FFT layout, PAW/spin configuration, scheduler, and filesystem behavior may
  require customized handling.
- Preserve the tutorial's validation principles instead of copying example values.

My context:

- Calculation: <CALCULATION_ROOT>
- QE build/source: <QE_VERSION_OR_SOURCE>
- Cluster: <CLUSTER>
- Prefix: <PREFIX>
- Objective: <OBJECTIVE>
- Current state: <KNOWN_STATE>
- Constraints: <CONSTRAINTS>

First explain how the tutorial maps to my calculation, including verified assumptions,
unknowns, and required customizations. Then collaborate with me on a tailored design.

Treat existing data as read-only. Prefer explicit manifests, provenance, dry runs, new
output paths, and focused tests. Never infer binary offsets or correctness from
filenames, irrep numbers, or file size alone, and state clearly when the exact QE I/O
behavior cannot be established.
```
