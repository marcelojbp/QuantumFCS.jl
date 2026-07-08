# [API](@id api)

## Computing cumulants

```@docs
QuantumFCS.fcscumulants_recursive
QuantumFCS.factorial_cumulants
```

## Problem types

```@docs
QuantumFCS.FCSProblem
QuantumFCS.LindbladFCS
QuantumFCS.PreparedLindbladFCS
QuantumFCS.prepare_fcs_context
```

## Steady state

Package-level helpers to solve for the trace-constrained steady state and reuse the
preconditioner built for that solve in the iterative FCS backend (see
[Preparing the steady state for iterative FCS](@ref steady-state-prep)):

```@docs
QuantumFCS.trace_constrained_steadystate
QuantumFCS.TraceConstrainedSteadyState
QuantumFCS.trace_constrained_system
QuantumFCS.TraceConstrainedSystem
QuantumFCS.shifted_ilu_preconditioner
```

## Drazin inverse helpers

The prepared Drazin solvers used internally are documented on the
[Drazin solvers](@ref solvers) page. The lower-level building blocks are:

```@docs
QuantumFCS.drazin
QuantumFCS.drazin_apply
QuantumFCS.m_jumps
```
