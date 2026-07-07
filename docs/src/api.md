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

## Drazin inverse helpers

The prepared Drazin solvers used internally are documented on the
[Drazin solvers](@ref solvers) page. The lower-level building blocks are:

```@docs
QuantumFCS.drazin
QuantumFCS.drazin_apply
QuantumFCS.m_jumps
```
