# ADR-0007: Glasses spike (D-07)

- Status: Accepted (settled default, spec v2); build order amended by [ADR-0059](0059-code-first-build-order.md): glasses-path code is built and tested now (Mac, simulation, JVM checks); claims about real glasses still wait for the hardware
- Date: 2026-10-28
- Source: docs/spec/ASB-Build-Spec-v2.pdf, section 15

## Decision

Skip until G1.

## Where it is enforced

Hardware/process decision; documented only (no code path).

## Changing it

Write a new ADR that supersedes this one, update CHANGELOG.md, and (if it touches a guardrail) get owner sign-off. A change to a frozen test threshold cannot be used to declare a pass retroactively.
