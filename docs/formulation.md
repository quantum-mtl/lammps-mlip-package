# Formulation note on MLIP

## Order Parameter

Consider a neighboring atomic density of *atom* {math}`i` between *element* {math}`s`, {math}`\rho^{(i, s)}(\mathbf{r})`, and expand it with basis functions,
```{math}
  \rho^{(i, s)}(\mathbf{r})
    =: \sum_{n=1}^{n_{\mathrm{max}}} \sum_{l=0}^{l \leq l_{\mathrm{max}}} \sum_{m=-l}^{l} a^{(i, s)}_{nlm} f_{n}(r) Y_{l}^{m}(\hat{\mathbf{r}}).
```
We refer to the coefficients {math}`a^{(i, s)}_{nlm}` as *order parameters*.

Let {math}`\mathcal{S}` be a set of pair elements [^notation],
```{math}
  \mathcal{S} = \left\{ [A, A], [A, B], [B, B], \dots \right\}.
```
It is useful to generalize an order parameter with a set of pair elements, {math}`[s, s'] \in \mathcal{S}`, as
```{math}
  a^{(i)}_{nlm, [s, s'] } :=
  \begin{cases}
    a^{(i, s'')}_{nlm} & (\exists s'' \,s.t.\, [s_{i}, s''] = [s, s'] ) \\
    0 & (\mathrm{otherwise})
  \end{cases},
```
where {math}`s_{i}` denotes a element type of atom {math}`i`.

[^notation]: A square bracket $[ \cdot ]$ indicate an unordered set.

In practice, the order parameters are calculated with a neighbor list within a given cut off radius {math}`r_{c}`,
```{math}
  a^{(i, s)}_{nlm} &= \sum_{ j \in \mathcal{N}_{i,s} } f_{n}(r_{ij}) Y_{l}^{m}(\hat{\mathbf{r}_{ij}})^{\ast}
  \quad \mbox{where} \quad
  \mathcal{N}_{i, s} := \left\{ j \mid r_{ij} \leq r_{c}, s_{j} = s \right\} \\
  \phi_{nlm}(\mathbf{r})
  &:= f_{n}(r) Y_{l}^{m}(\hat{\mathbf{r}})^{\ast}.
```
Note that radial basis function {math}`f_{n}(\cdot)` is a real function.

## Structural Feature
Consider {math}`q`th-order polynomial of {math}`\{ a^{(i)}_{nlm, t}\}_{n, l, m, t \in \mathcal{S}}` which is invariant with {math}`O(3)` actions.
```{math}
  \mathcal{L}_{q}
  &:= \left\{ ( [l_{1}, \dots, l_{q}], \sigma) \mid l_{i} = 0, \dots, l_{\max}^{(q)} (i = 1, \dots, q), \mathbf{C}^{ l_{1} \dots l_{q}, \sigma } \neq \mathbf{0}, \sum_{i=0}^{q} l_{i} \, \mbox{is even} \right\} \\
  \mathcal{L}
  &:= \bigcup_{q = 1}^{q_{\max}} \mathcal{L}_{q}
```

*angular-element pairs*
```{math}
  \mathcal{K}_{q}
  &= \left\{ ( [l_{1}, \dots, l_{q}], [t_{1}, \dots, t_{q}], \sigma ) \mid ( [l_{1}, \dots, l_{q}], \sigma) \in \mathcal{L}_{q}, t_[1], \dots, t_{q} \in \mathcal{S}, t_{1} \cap \dots \cap t_{q} \neq \varnothing \right\} \\
  \mathcal{K}
  &:= \bigcup_{q=1}^{q_{\max}} \mathcal{K}_{q}
```
Note that regardless of element type of atom {math}`i`, when {math}`t_{1} \cap \dots \cap t_{q} = \varnothing`, the structural feature becomes zero.

*Structural feature*
```{math}
  d^{(i)}_{ n, \mathbf{k} }
  &:= \sum_{m_{1}=-l_{1}}^{l_{1}} \dots \sum_{m_{q}=-l_{q}}^{l_{q}} C^{l_{1} \dots l_{q}, \sigma_{\mathbf{k}} }_{m_{1} \dots m_{q}} a^{(i)}_{nl_{1}m_{1}, t_{1}} \dots a^{(i)}_{n l_{q} m_{q}, t_{q}} \\
  &\quad \mbox{where} \quad \mathbf{k} = ( [l_{1}, \dots, l_{q}], [t_{1}, \dots, t_{q}], \sigma ) \in \mathcal{K}_{q}.
```
The index {math}`\sigma_{\mathbf{k}}` distinguishes several Irreps with the same unordered set of angular numbers {math}`[l_{1}, \dots, l_{q}]`.
We refer the tuple {math}`(n, \mathbf{k}, \sigma_{k})` as a *feature index*.

## Polynomial Feature

*elements intersection*
```{math}
  e(\mathbf{k}) := t_{1} \cap \dots \cap t_{q}
  \quad \mbox{where} \quad \mathbf{k} = ( \cdots, [t_{1}, \dots, t_{q}], \cdot ) \in \mathcal{K}_{q}.
```

*effective feature indices families*
```{math}
  \overline{\mathcal{P}}_{p}
  := \left\{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \mid \mathbf{k}_{1}, \dots, \mathbf{k}_{p} \in \mathcal{K}, e(\mathbf{k}_{1}) \cap \dots \cap e(\mathbf{k}_{p}) \neq \varnothing \right\}
```

*polynomial feature*
```{math}
  d^{(i)}_{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] }
  &:= \prod_{ \mu = 1 }^{ p } d^{(i)}_{ n_{\mu} \mathbf{k}_{\mu} } \nonumber \\
  &\mbox{where} \quad
    [(n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p})] \in \overline{\mathcal{P}}_{p}
```

## Potential Energy Model

```{math}
  E(\{ \mathbf{r}_{i} \})
  &= \sum_{i=1}^{N} F \left( \{ d^{(i)}_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] } \}_{[\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] \in \overline{\mathcal{P}}_{p}}^{p=1,\dots} \right) \\
  F \left( \{ d^{(i)}_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] } \}_{[\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] \in \overline{\mathcal{P}}_{p}}^{p=1,\dots} \right)
  &= \sum_{p \geq 1} \sum_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] \in \overline{\mathcal{P}}_{p} } w_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] } d^{(i)}_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] }
```
Note that the weight {math}`w_{ [\mathbf{f}_{1}, \dots, \mathbf{f}_{p}] }` does not depend on an element type of atom {math}`i`.

```
\begin{figure}
  \begin{algorithm}[H]
    \caption{Compute energy $E$}
    \begin{algorithmic}
      \Require $ \{ d^{(i)}_{ n, \mathbf{k} } \}_{ i \in [N] ; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} } $
      \State $E \gets 0$
      \For{ $ i \in [N]$ }
        \State $E_{i} \gets 0$
        \For{ $[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \in \mathcal{P}$ }
          \If {$s_{i} \notin e(\mathbf{k}_{1}) \cap \dots \cap e(\mathbf{k}_{p}) $}
            \State \textbf{continue}
          \EndIf
          \State $E_{i} \gets E_{i} + w_{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ]} \cdot \prod_{\mu=1}^{p} d^{(i)}_{ n_{\mu}, \mathbf{k}_{\mu} }$
        \EndFor
        \State $E \gets E + E_{i}$
      \EndFor
      \State \textbf{return} $E$
    \end{algorithmic}
  \end{algorithm}
  % \caption{caption}
\end{figure}
```

## Derivative

### Forces and Stress Tensor

% pairwise forces
```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \mathbf{F}_{ji} := - \pdev{E^{(i)}}{\mathbf{r}_{j}}
    &= - \sum_{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \in \mathcal{P} } w_{[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ]} \sum_{\nu = 1}^{p} \pdev{ d^{(i)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{j} } \prod_{\nu' \neq \nu} d^{(i)}_{ n_{\nu'} \mathbf{k}_{\nu'} } \nonumber \\
    &=: -\sum_{n \in [n_{\max}]} \sum_{\mathbf{k} \in \mathcal{K}} \pdev{ d^{(i)}_{n, \mathbf{k}} }{\mathbf{r}_{j}} H^{(i)}_{n, \mathbf{k}} \\
    &= -\sum_{n \in [n_{\max}]} \sum_{ (\mathbf{l}, \mathbf{t}, \sigma) \in \mathcal{K}} \sum_{\mathbf{m} \in \mathcal{M}_{\mathbf{l}}} C^{\mathbf{l} \sigma}_{\mathbf{m}} \sum_{\mu = 1}^{|\mathbf{l}|} \delta_{ t_{\mu}, \{ s_{i}, s_{j} \} } \pdev{ \phi_{n l_{\mu} m_{\mu} } }{ \mathbf{r} }(\mathbf{r}_{ij}) \prod_{ \mu' \neq \mu } a^{(i)}_{ n l_{\mu'} m_{\mu'}, t_{\mu'} } \nonumber \\
    &=: -\sum_{n \in [n_{\max}]} \sum_{l = 0}^{l_{\max}} \sum_{m = -l}^{l} \pdev{ \phi_{nlm} }{ \mathbf{r} }(\mathbf{r}_{ij}) G^{(i)}_{ nlm, \{ s_{i}, s_{j} \} } \\
  \mathbf{F}_{ij}
    = - \pdev{E^{(j)}}{\mathbf{r}_{i}}
    &= \sum_{n \in [n_{\max}]} \sum_{l = 0}^{l_{\max}} (-)^{l} \sum_{m = -l}^{l} \pdev{ \phi_{nlm} }{ \mathbf{r} }(\mathbf{r}_{ij}) G^{(j)}_{ nlm, \{ s_{i}, s_{j} \} }
```

% forces for each atom
```{math}
  \mathbf{F}_{i}
  &:= \sum_{j \in [N]} \mathbf{F}_{ij}
  = \mathbf{F}_{ii} + \sum_{j \in \mathcal{N}_{i}} \mathbf{F}_{ij} \nonumber \\
  &= \sum_{j \in \mathcal{N}_{i}} \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right)
```

% stress tensor
```{math}
  \mathbf{\sigma}
  &:= -\frac{1}{2} \sum_{i \in [N]} \mathbf{r}_{i} \otimes \mathbf{F}_{i} \\
  &= -\frac{1}{2} \sum_{i \in [N]} \sum_{j \in \mathcal{N}_{i}} \mathbf{r}_{i} \otimes \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right) \nonumber \\
  &= - \sum_{i \in [N]} \sum_{j \in \mathcal{N}_{i}^{\mathrm{half}}} \left( \mathbf{r}_{i} \otimes \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right) + \mathbf{r}_{j} \otimes \left( \mathbf{F}_{ji} - \mathbf{F}_{ij} \right) \right) \nonumber \\
  &= - \sum_{i \in [N]} \sum_{j \in \mathcal{N}_{i}^{\mathrm{half}}} \mathbf{r}_{ij} \otimes \left( \mathbf{F}_{ij} - \mathbf{F}_{ji} \right)
```

```
\begin{figure}
  \begin{algorithm}[H]
    \caption{
      Compute adjoint $\{ H^{(i)}_{n, \mathbf{k}} \}_{i \in [N]; n \leq n_{\max}; \mathbf{k} \in \mathcal{K}}$
    }
    \begin{algorithmic}
      \Require $ \{ d^{(i)}_{ n, \mathbf{k} } \}_{ i \in [N] ; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} } $
      \For {$i \in [N]$}
        \For {$ n \in [n_{\max}], \mathbf{k} \in \mathcal{K}$}
          \State $H^{(i)}_{n, \mathbf{k}} \gets 0$
        \EndFor
      \EndFor
      \For {$i \in [N]$}
        \For{ $[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \in \mathcal{P}$ }
          \For {$\nu \in [p]$}
            \State $H^{(i)}_{n_{\nu}, \mathbf{k}_{\nu}} \gets H^{(i)}_{n_{\nu}, \mathbf{k}_{\nu}} + w_{[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ]} \prod_{\nu' \neq \nu} d^{(i)}_{ n_{\nu'} \mathbf{k}_{\nu'} } $
          \EndFor
        \EndFor
      \EndFor
      \State \textbf{return} $\{ H^{(i)}_{n, \mathbf{k}} \}_{i \in [N]; n \leq n_{\max}; \mathbf{k} \in \mathcal{K}}$
    \end{algorithmic}
  \end{algorithm}
\end{figure}
```

```
\begin{figure}
  \begin{algorithm}[H]
    \caption{
      Compute adjoints $\{ G^{(i)}_{nlm, t} \}_{ i \in [N]; n \in [n_{\max}]; l \leq l_{\max}; |m| \leq l; t \in \mathcal{S} }$
    }
    \begin{algorithmic}
      \Require $ \{ H^{(i)}_{n, \mathbf{k}} \}_{i \in [N]; n \leq n_{\max}; \mathbf{k} \in \mathcal{K}} $
      \For { $i \in [N]$ }
        \For { $ t \in \mathcal{S}$ }
          \For { $ n \in [n_{\max}]$ }
            \For {$ l \leq l_{\max}, |m| \leq l$}
              \State $ G^{(i)}_{nlm, t} \gets 0$
            \EndFor
          \EndFor
        \EndFor
      \EndFor

      \For { $i \in [N]$ }
        \For { $(\mathbf{l}, \mathbf{t}, \sigma) \in \mathcal{K}$ }
          \If { $s_{i} \notin e(\mathbf{t})$ }
            \State \textbf{continue}
          \EndIf
          \State $q \gets |\mathbf{l}|$
          \For { $ n \in [n_{\max}]$ }
            \For { $\mathbf{m} \in \mathcal{M}_{\mathbf{l}}$ }
              \For { $\mu \in [q]$ }
                \State $G^{(i)}_{ n l_{\mu} m_{\mu}, t_{\mu}} \gets G^{(i)}_{ n l_{\mu} m_{\mu}, t_{\mu}} + H^{(i)}_{n, (\mathbf{l}, \mathbf{t}, \sigma) } C^{\mathbf{l}, \sigma}_{\mathbf{m}} \prod_{\mu' \neq \mu} a^{(i)}_{ n l_{\mu'} m_{\mu'}, t_{\mu'}} $
              \EndFor
            \EndFor
          \EndFor
        \EndFor
      \EndFor
      \State \textbf{return} $\{ G^{(i)}_{nlm, t} \}_{ i \in [N]; n \in [n_{\max}]; l \leq l_{\max}; |m| \leq l; t \in \mathcal{S} }$
    \end{algorithmic}
  \end{algorithm}
\end{figure}
```

```
\begin{figure}
  \begin{algorithm}[H]
    \caption{Compute forces $\{ \mathbf{F}_{i} \}_{i \in [N]}$ and $\mathbf{\sigma}$}
    \begin{algorithmic}
      \Require $ \{ d^{(i)}_{ n, \mathbf{k} } \}_{ i \in [N] ; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} } $
      %
      \For{ $ i \in [N]$ }
        \State $\mathbf{F}_{i} \gets \mathbf{0}$
      \EndFor
      \State $\mathbf{\sigma} \gets O$
      %
      \State compute $\{ H^{(i)}_{n, \mathbf{k}} \}_{ i \in [N] ; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} }$
      \State compute $\{ G^{(i)}_{nlm, t} \}_{ i \in [N]; n \in [n_{\max}]; l \leq l_{\max}; |m| \leq l; t \in \mathcal{S} }$
      \For{ $ i \in [N]$ }
        \For{$j \in \mathcal{N}_{i}^{\mathrm{half}}$}
          \State $\mathbf{F}_{ji} \gets \mathbf{0}$
          \State $\mathbf{F}_{ij} \gets \mathbf{0}$
          \State compute $ \{ \pdev{ \phi_{nlm} }{\mathbf{r}}(\mathbf{r}_{ij}) \}_{nlm}$
          \For { $ n \in [n_{\max}]$}
            \For {$ l \leq l_{\max} $}
              \State $\mathbf{F}_{ji} \gets \mathbf{F}_{ji} - \re G^{(i)}_{nl0, \{ s_{i}, s_{j} \}} \re \pdev{ \phi_{nl0} }{\mathbf{r}}(\mathbf{r}_{ij}) - 2 \sum_{m > 0} G^{(i)}_{nlm, \{ s_{i}, s_{j} \}} \pdev{ \phi_{nlm} }{\mathbf{r}}(\mathbf{r}_{ij})$
              \State $\mathbf{F}_{ji} \gets \mathbf{F}_{ji} + (-)^{l} \left( \re G^{(i)}_{nl0, \{ s_{i}, s_{j} \}} \re \pdev{ \phi_{nl0} }{\mathbf{r}}(\mathbf{r}_{ij}) + 2 \sum_{m > 0} G^{(i)}_{nlm, \{ s_{i}, s_{j} \}} \pdev{ \phi_{nlm} }{\mathbf{r}}(\mathbf{r}_{ij}) \right)$
            \EndFor
          \EndFor
          \State $\mathbf{f} \gets \mathbf{F}_{ij} - \mathbf{F}_{ji}$
          \State $\mathbf{F}_{i} \gets \mathbf{F}_{i} + \mathbf{f}$
          \State $\mathbf{F}_{j} \gets \mathbf{F}_{j} - \mathbf{f}$
          \State $\mathbf{\sigma} \gets \mathbf{\sigma} - \mathbf{r}_{ij} \otimes \mathbf{f}$
        \EndFor
      \EndFor
      \State \textbf{return} $\{ \mathbf{F}_{i} \}_{i \in [N]}$ and $\mathbf{\sigma}$
    \end{algorithmic}
  \end{algorithm}
  % \caption{caption}
\end{figure}
```

```{math}
  \newcommand{\ddev}[2]{\frac{d #1}{d #2}}

  \pdev{ \phi_{nlm}(\mathbf{r}) }{ r_{\alpha} }
  &= \left( \ddev{ f_{n} }{r}(r) \frac{r_{\alpha}}{r} Y_{l}^{m}(\hat{\mathbf{r}}) + f_{n}(\mathbf{r}) \pdev{ Y_{l}^{m}(\hat{\mathbf{r}}) }{r_{\alpha}} \right)^{\ast} \\
  \pdev{ Y_{l}^{m} }{r_{x}}
    &= \frac{\cos \theta \cos \phi}{r} \pdev{ Y_{l}^{m} }{\theta} - \frac{ i m \sin \phi}{r \sin \theta} Y_{l}^{m} \\
  \pdev{ Y_{l}^{m} }{r_{y}}
    &= \frac{\cos \theta \sin \phi}{r} \pdev{ Y_{l}^{m} }{\theta} + \frac{ i m \cos \phi}{r \sin \theta} Y_{l}^{m} \\
  \pdev{ Y_{l}^{m} }{r_{z}}
    &= -\frac{\sin \theta}{r} \pdev{ Y_{l}^{m} }{\theta}
```

Parity symmetry
```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \pdev{ \phi_{nl-m} }{\mathbf{r}}
    &= (-)^{m} \pdev{ \phi_{nlm}^{\ast} }{\mathbf{r}} \\
  G^{(i)}_{nl-m, t}
    &= (-)^{m} G^{(i) \ast}_{nlm, t} \quad (\because C^{\mathbf{l}, \sigma}_{\mathbf{m}} \neq 0 \Rightarrow \sum_{\mu} m_{\mu} = 0 )
```

## Force Features

```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \mathbf{F}_{i}
    &= \sum_{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \in \mathcal{P} }
          w_{[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ]}
          \sum_{j \in \mathcal{N}_{i}} \sum_{\nu = 1}^{p}
          \left(
            \pdev{ d^{(i)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{j} } \prod_{\nu' \neq \nu} d^{(i)}_{ n_{\nu'} \mathbf{k}_{\nu'} }
            - \pdev{ d^{(j)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{i} } \prod_{\nu' \neq \nu} d^{(j)}_{ n_{\nu'} \mathbf{k}_{\nu'} }
          \right) \\
    &=: \sum_{ [ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ] \in \mathcal{P} }
          w_{[ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) ]}
          \mathbf{D}^{(i)}_{ (n_{1}, \mathbf{k}_{1}), \dots, (n_{p}, \mathbf{k}_{p}) }
```

```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \pdev{ d^{(i)}_{ n, (\mathbf{l}, \mathbf{t}, \sigma) } }{ \mathbf{r}_{j}}
    &= \sum_{\mathbf{m} \in \mathcal{M}_{\mathbf{l}}} C^{\mathbf{l} \sigma}_{\mathbf{m}} \sum_{\mu = 1}^{|\mathbf{l}|}
          \delta_{ t_{\mu}, \{ s_{i}, s_{j} \} } \pdev{ \phi_{n l_{\mu} m_{\mu} } }{ \mathbf{r} }(\mathbf{r}_{ij}) \prod_{ \mu' \neq \mu } a^{(i)}_{ n l_{\mu'} m_{\mu'}, t_{\mu'} } \\
  \pdev{ d^{(j)}_{ n, (\mathbf{l}, \mathbf{t}, \sigma) } }{ \mathbf{r}_{i}}
    &= -\sum_{\mathbf{m} \in \mathcal{M}_{\mathbf{l}}} C^{\mathbf{l} \sigma}_{\mathbf{m}} \sum_{\mu = 1}^{|\mathbf{l}|}
          \delta_{ t_{\mu}, \{ s_{i}, s_{j} \} } (-)^{l_{\mu}} \pdev{ \phi_{n l_{\mu} m_{\mu} } }{ \mathbf{r} }(\mathbf{r}_{ij}) \prod_{ \mu' \neq \mu } a^{(j)}_{ n l_{\mu'} m_{\mu'}, t_{\mu'} } \\
```

```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \mathbf{\sigma}
    &= \sum_{i=1}^{N} \sum_{ [f_{1}, \dots, f_{p}] \in \mathcal{P} } w_{[f_{1}, \dots, f_{p}]}
      \frac{1}{2}
      \mathbf{r}_{i} \otimes
      \left(
          \pdev{ d^{(j)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{i} } \prod_{\nu' \neq \nu} d^{(j)}_{ n_{\nu'} \mathbf{k}_{\nu'} }
          - \pdev{ d^{(i)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{j} } \prod_{\nu' \neq \nu} d^{(i)}_{ n_{\nu'} \mathbf{k}_{\nu'} }
      \right) \\
    &=: \sum_{ [f_{1}, \dots, f_{p}] \in \mathcal{P} } w_{[f_{1}, \dots, f_{p}]} B_{f_{1}, \dots, f_{p}}
```

```
\begin{figure}
  \begin{algorithm}[H]
    \caption{Compute force features $\{ \mathbf{D}^{(i)}_{ f_{1}, \dots, f_{p} } \}_{i \in [N]; [f_{1}, \dots, f_{p}] \in \mathcal{P} }$ and stress features $\{ \mathbf{B}_{ f_{1}, \dots, f_{p} } \}_{ [f_{1}, \dots, f_{p}] \in \mathcal{P} }$ }
    \begin{algorithmic}
      \Require $ \{ d^{(i)}_{ n, \mathbf{k} } \}_{ i \in [N] ; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} } $
      \State compute $\{ \pdev{ d^{(i)}_{ n, \mathbf{k} } }{ \mathbf{r}_{j}}, \pdev{ d^{(j)}_{ n, \mathbf{k} } }{ \mathbf{r}_{i} } \}_{i \in [N]; j \in \mathcal{N}_{i}^{\mathrm{half}}; n \in [n_{\max}]; \mathbf{k} \in \mathcal{K} }$
      \For{ $ i \in [N]$ }
        \For {$j \in \mathcal{N}_{i}^{\mathrm{half}}$}
          \For {$[f_{1}, \dots, f_{p}] \in \mathcal{P}$}
            \For {$\nu = 1, \dots, p$}
              \State $\mathbf{g} \gets \pdev{ d^{(i)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{j} } \prod_{\nu' \neq \nu} d^{(i)}_{ n_{\nu'} \mathbf{k}_{\nu'} } - \pdev{ d^{(j)}_{ n_{\nu}, \mathbf{k}_{\nu} } }{ \mathbf{r}_{i} } \prod_{\nu' \neq \nu} d^{(j)}_{ n_{\nu'} \mathbf{k}_{\nu'} } $
              \State $\mathbf{D}^{(i)}_{ f_{1}, \dots, f_{p} } \pluseq \mathbf{g}$
              \State $\mathbf{D}^{(j)}_{ f_{1}, \dots, f_{p} } \minuseq \mathbf{g}$
              \State $\mathbf{\sigma} \minuseq \mathbf{r}_{ij} \otimes \mathbf{g}$
            \EndFor
          \EndFor
        \EndFor
      \EndFor
      \State \textbf{return} $\{ \mathbf{D}^{(i)}_{ f_{1}, \dots, f_{p} } \}_{i \in [N]; [f_{1}, \dots, f_{p}] \in \mathcal{P} }, \{ \mathbf{B}_{ f_{1}, \dots, f_{p} } \}_{ [f_{1}, \dots, f_{p}] \in \mathcal{P} }$
    \end{algorithmic}
  \end{algorithm}
  % \caption{caption}
\end{figure}
```

## Spherical harmonics

With {math}`l, m` integers such that {math}`0 \leq m \leq l`, and {math}`0 \leq \theta \leq \pi`,
```{math}
  Y_{l}^{m}(\theta, \phi)
    &:= \sqrt{ \frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!} } e^{im\phi} P_{l}^{(m)}(\cos \theta) \\
  Y_{l}^{-m}(\theta, \phi)
    &:= (-)^{m} Y_{l}^{m}(\theta, \phi)^{\ast}
```
This definition adopts Condon-Shortley phase.

Recursion for computing {math}`Y_{l}^{m}` {cite}`1410.1748`
```{math}
  Y_{l}^{m}(\theta, \phi) &=: \frac{1}{\sqrt{2}} e^{im\phi} \bar{P}_{l}^{m} \\
  \bar{P}_{0}^{0} &= \sqrt{\frac{1}{2 \pi}} \\
  \bar{P}_{m}^{m} &= -\sin \theta \sqrt{1 + \frac{1}{2m}} \bar{P}_{m-1}^{m-1} \quad (m \geq 1) \\
  \bar{P}_{m+1}^{m} &= \sqrt{2m+3} \cos \theta \bar{P}_{m}^{m} \quad (m \geq 0) \\
  a_{l}^{m} &:= \sqrt{ \frac{4l^{2} -1}{l^{2}-m^{2}} } \\
  b_{l}^{m} &:= -\sqrt{ \frac{ (l-1)^{2} - m^{2} }{ 4(l-1)^{2} - 1 } } \\
  \bar{P}_{l}^{m} &= a_{l}^{m} ( \cos \theta \bar{P}_{l-1}^{m} + b_{l}^{m} \bar{P}_{l-2}^{m} ) \quad ( l \geq 2, 0 \leq m \leq l - 2)
```

```{math}
  Q_{l}^{m}
    &:= \frac{ \bar{P}_{l}^{m} }{ \sin \theta } \quad (l \geq 1, 0 < m \leq l) \\
  Q_{1}^{1} &= -\sqrt{\frac{3}{4 \pi}} \\
  Q_{m}^{m} &= -\sin \theta \sqrt{1 + \frac{1}{2m}} Q_{m-1}^{m-1} \quad (m \geq 1) \\
  Q_{m+1}^{m} &= \sqrt{2m+3} \cos \theta Q_{m}^{m} \quad (m \geq 0) \\
  Q_{l}^{m} &= a_{l}^{m} ( \cos \theta Q_{l-1}^{m} + b_{l}^{m} Q_{l-2}^{m} ) \quad ( l \geq 2, 0 \leq m \leq l - 2)
```

```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \begin{pmatrix}
    \pdev{Y_{l}^{m}}{r_{x}} \\
    \pdev{Y_{l}^{m}}{r_{y}} \\
    \pdev{Y_{l}^{m}}{r_{z}} \\
  \end{pmatrix}
    &= \frac{ e^{im\phi} }{ \sqrt{2} r } \left[
      ( m \cos \theta Q_{l}^{m} + \sqrt{ (l-m)(l+m+1) } \bar{P}_{l}^{m+1} )
      \begin{pmatrix}
        \cos \theta \cos \phi \\
        \cos \theta \sin \phi \\
        \sin \theta \\
      \end{pmatrix}
      + im Q_{l}^{m}
      \begin{pmatrix}
        -\sin \phi \\
        \cos \phi \\
        0 \\
      \end{pmatrix}
    \right]
```

```{math}
  \newcommand{\pdev}[2]{\frac{\partial #1}{\partial #2}}

  \pdev{ Y_{lm}^{\ast}(\theta, \phi) }{ \theta }
    &= m \cot \theta Y_{lm}^{\ast}(\theta, \phi) + \sqrt{ (l-m)(l+m+1) } e^{i\phi} Y_{l(m+1)}^{\ast}(\theta, \phi) \\
```

## References

```{bibliography}
:filter: docname in docnames
```
