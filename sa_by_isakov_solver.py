import dimod
import os
import subprocess
import ast


class SaByIsakovSolver(dimod.Sampler):
    SIMULATED_ANNEALER_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'siman', 'an_ss_ge_fi_vdeg'
    )
    SIMULATED_ANNEALER_PATH_OMP = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'siman', 'an_ss_ge_fi_vdeg_omp'
    )
    INSTANCES_DIRECTORY = 'instances'

    def __init__(self, simulated_annealer_path=None, simulated_annealer_path_omp=None,
                 instances_directory=None, num_sweeps=2000, num_rep=1000, num_threads=None,
                 beta0=0.1, beta1=3.0, name=None, show_timer=False):

        self.num_sweeps = num_sweeps
        self.num_rep = num_rep
        self.num_threads = num_threads
        self.beta0 = beta0
        self.beta1 = beta1
        self.name = name if name is not None else str(id(bqm))
        self.show_timer = show_timer

        if simulated_annealer_path is not None:
            self.simulated_annealer_path = simulated_annealer_path
        else:
            self.simulated_annealer_path = self.SIMULATED_ANNEALER_PATH

        if simulated_annealer_path_omp is not None:
            self.simulated_annealer_path_omp = simulated_annealer_path_omp
        else:
            self.simulated_annealer_path_omp = self.SIMULATED_ANNEALER_PATH_OMP

        if instances_directory is not None:
            self.instances_directory = instances_directory
        else:
            self.instances_directory = self.INSTANCES_DIRECTORY

    def properties(self):
        pass

    def parameters(self):
        pass

    def sample(self, bqm: dimod.BinaryQuadraticModel, name=None, num_sweeps=None,
               num_rep=None, beta0=None, beta1=None, show_timer=None, num_threads=None):
        name = name if name is not None else self.name
        num_sweeps = num_sweeps if num_sweeps is not None else self.num_sweeps
        num_rep = num_rep if num_rep is not None else self.num_rep
        beta0 = beta0 if beta0 is not None else self.beta0
        beta1 = beta1 if beta1 is not None else self.beta1
        show_timer = show_timer if show_timer is not None else self.show_timer
        num_threads = num_threads if num_threads is not None else self.num_threads

        vartype = bqm.vartype
        bqm.change_vartype(dimod.SPIN)

        mask_to_sa = {it: i for i, it in enumerate(bqm.variables)}
        mask_from_sa = {i: it for i, it in enumerate(bqm.variables)}
        bqm.relabel_variables(mask_to_sa)

        if not os.path.exists(self.instances_directory):
            os.mkdir(self.instances_directory)

        filename = f"{name}.txt"
        path = os.path.join(self.instances_directory, filename)
        with open(path, 'w') as pf:
            pf.write(name + '\n')
            bqm.to_coo(pf)

        command = [self.simulated_annealer_path, '-l', path, '-s', str(num_sweeps),
                   '-r', str(num_rep), '-g', '-b0', str(beta0), '-b1', str(beta1)]
        if num_threads is not None:
            command[0] = self.simulated_annealer_path_omp
            command.extend(['-t', str(num_threads)])

        output = subprocess.run(
            command, stdout=subprocess.PIPE
        ).stdout.decode("utf-8").strip('\n')

        energy = float(next(filter(lambda a: a != '', output.split('\n')[0].split(' '))))
        energy += bqm.offset

        configurations = []
        for line in output.split('\n')[1:]:
            try:
                configuration = ast.literal_eval(line)
            except IndentationError:
                continue
            configurations.append({mask_from_sa[it]: configuration[it] if configuration[it] > 0 else 0
                                   for it in mask_from_sa})

        response = dimod.Response.from_samples(configurations, {
            'energy': [energy]*len(configurations),
            'num_occurrences': [num_rep/len(configurations)]*len(configurations)  # TODO
        }, {}, vartype)
        return response

if __name__ == '__main__':
    bqm = dimod.BinaryQuadraticModel({}, {}, 0, dimod.BINARY)
    # bqm.add_interaction('a', 'b', -1)

    bqm.add_variable('a', 1)
    bqm.add_variable('b', 1)
    bqm.add_interaction('a', 'b', -2)

    solver = SaByIsakovSolver()
    print(solver.sample(bqm, name='test'))
