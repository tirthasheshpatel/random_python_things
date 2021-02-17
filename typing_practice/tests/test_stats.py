from ..stats import mean

mean([1.], weights=[1.], axis=1.)
mean([1.], weights=[1.], axis="this_will_piss_mypy_off")
