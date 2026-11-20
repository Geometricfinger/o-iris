# ADR-0009: Platform (D-09)

- Status: Accepted (settled default, spec v2); target and build order amended by [ADR-0059](0059-code-first-build-order.md): Android smart glasses first, then possibly a helmet; code infrastructure first; the Mac and the rig stay test benches
- Date: 2026-10-28
- Source: docs/spec/ASB-Build-Spec-v2.pdf, section 15

## Decision

Rig + Android operator app; glasses Phase 2 only after G1.

## Where it is enforced

Hardware/process decision; documented only (no code path).

## Changing it

Write a new ADR that supersedes this one, update CHANGELOG.md, and (if it touches a guardrail) get owner sign-off. A change to a frozen test threshold cannot be used to declare a pass retroactively.
