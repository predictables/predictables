from ._distn_infrastructure import rv_discrete
from typing import Any

class binom_gen(rv_discrete): ...

binom: Any

class bernoulli_gen(binom_gen): ...

bernoulli: Any

class betabinom_gen(rv_discrete): ...

betabinom: Any

class nbinom_gen(rv_discrete): ...

nbinom: Any

class geom_gen(rv_discrete): ...

geom: Any

class hypergeom_gen(rv_discrete): ...

hypergeom: Any

class nhypergeom_gen(rv_discrete): ...

nhypergeom: Any

class logser_gen(rv_discrete): ...

logser: Any

class poisson_gen(rv_discrete): ...

poisson: Any

class planck_gen(rv_discrete): ...

planck: Any

class boltzmann_gen(rv_discrete): ...

boltzmann: Any

class randint_gen(rv_discrete): ...

randint: Any

class zipf_gen(rv_discrete): ...

zipf: Any

class zipfian_gen(rv_discrete): ...

zipfian: Any

class dlaplace_gen(rv_discrete): ...

dlaplace: Any

class skellam_gen(rv_discrete): ...

skellam: Any

class yulesimon_gen(rv_discrete): ...

yulesimon: Any

class _nchypergeom_gen(rv_discrete):
    rvs_name: Any
    dist: Any

class nchypergeom_fisher_gen(_nchypergeom_gen):
    rvs_name: str
    dist: Any

nchypergeom_fisher: Any

class nchypergeom_wallenius_gen(_nchypergeom_gen):
    rvs_name: str
    dist: Any

nchypergeom_wallenius: Any
