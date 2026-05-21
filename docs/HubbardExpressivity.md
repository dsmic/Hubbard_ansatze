## Analyzing the expressivity of different wave function approaches with infidelity and energy optimization

We investigate the Hubbard model on a 4x4 lattice with periodic boundary conditions and 7 up and 7 down electrons. This setup is near the discussed high temperature superconductivity in the cuprates.

This is one of the largest systems where exact solutions are easily available, and different approaches result in comparable low energies. We investigate the hidden fermion and the Pfaffian approach, which both lead to energies about 1% above the exact ground-state energy (using about 100k parameters)

![Training](../imgs/Training.png)

As the energy does not seem to show much difference in the expressivity of the approaches, we test, how well the exact ground state can be expressed by these approaches. This is done by minimizing the infidelity with respect to an exact ground state (it is degenerated) and monitoring both the infidelity and the energy. After some optimizations we switch back to training the energy, while still monitoring. This allows us to plot energy versus infidelity, which shows that the Pfaffian can express the ground state much better (infidelity about 0.04) than the hidden fermion approach (infidelity about 0.15). 

![Expressivity Hidden Fermion](../imgs/ExpressivityHiddenFermion.png)
![Expressivity Pfaffian](../imgs/ExpressivityPfaffian.png)

We provide the source code for all calculations. Have a look at the GitHub repository:
[https://github.com/dsmic/Hubbard_ansatze](https://github.com/dsmic/Hubbard_ansatze).

Even if you are not interested in these expressivity investigations the source code might help to reproduce some of the recent results cited in the repository.
