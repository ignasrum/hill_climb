import numpy
from numpy.random import randn, rand


class HillClimb:
    def __init__(self, fitness, generator, var_count, initial_state, seed=0):
        self.fitness = fitness
        self.generator = generator
        self.var_count = var_count
        self.initial_state = initial_state
        numpy.random.seed(seed)

    def run(self, n_iterations=1000):
        solution = self.initial_state
        solution_eval = self.fitness(solution)
        solutions = list()
        solutions.append(solution)
        for i in range(n_iterations):
            candidate = self.generator(solution)
            candidate_eval = self.fitness(candidate)
            if candidate_eval <= solution_eval:
                solution, solution_eval = candidate, candidate_eval
                solutions.append(solution)
        return [solution, solution_eval, solutions]


# objective/fitness function
# takes args with arg_count of variables/coefficients
# return a single int/float
def fitness(solution):
    return numpy.sum((solution[0]**2.0)**(solution[1]**2.0))

# solution with correct number of variables as input
# utilize some random generation function + a step variable
# return one candidate solution
def generator(solution):
    step = 0.005
    candidate = []
    for arg in range(2):
        candidate.append(solution[arg] + randn() * step)
    candidate = numpy.asarray(candidate)
    return candidate

if __name__ == "__main__":
    hc = HillClimb(fitness, generator, 2, rand(2))
    [solution, solution_eval, solutions] = hc.run()

    for i, candidate in enumerate(solutions):
        print('>%d f(%s) = %.5f' % (i, candidate, fitness(candidate)))

    print("------------------")
    print('Final solution: >f(%s) = %f' % (solution, solution_eval))
