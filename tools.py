from dwavebinarycsp import stitch, irreducible_components, Constraint
import operator
import dwavebinarycsp
import dimod
from collections import Counter
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler          
from dwave.system import EmbeddingComposite
from minorminer import find_embedding
from dwave.cloud import Client
import pandas as pd
import numpy as np 
client = Client.from_config(token = '')
client.get_solvers()


def squeeze_same_results(results):
    squeezed_results = [results[0]]
    for item in results[1:]:
        for squeezed_item in squeezed_results:
            if item['results'] == squeezed_item['results']:
                if item['min_energy'] < squeezed_item['min_energy']:
                    squeezed_item['min_energy'] = item['min_energy']
                    squeezed_item['occurrences'] = item['occurrences']
                elif item['min_energy'] == squeezed_item['min_energy']:
                    squeezed_item['occurrences'] += item['occurrences']
                break
        else:
            squeezed_results.append(item)
    return squeezed_results


def parse_result(response):
    parsed_response = {}
    for datum in response.data(['sample', 'energy', 'num_occurrences']):
        key = (tuple(dict(datum.sample).items()), float(datum.energy))
        if key in parsed_response:
            parsed_response[key] = (datum.sample, parsed_response[key][1] + datum.num_occurrences)
        else:
            parsed_response[key] = (datum.sample, datum.num_occurrences)

    num_runs = sum([parsed_response[key][1] for key in parsed_response])
    results = []
    for key in parsed_response:
        results.append({
            'results': parsed_response[key][0],
            'min_energy': key[1],
            'occurrences': parsed_response[key][1] / num_runs * 100
        })
    return squeeze_same_results(results)


def get_response(response, embedding=None, qubits=None):
    results = parse_result(response)
    if embedding is not None:
        embedding_results = []
        for item in results:
            embedding_item = {
                'results': {},
                'min_energy': item['min_energy'],
                'occurrences': item['occurrences']
            }
            for A in embedding:
                embedding_item['results'][A] = sum([item['results'][qubit] for qubit in embedding[A]])
                embedding_item['results'][A] /= len(embedding[A])
                embedding_item['results'][A] = round(embedding_item['results'][A], 2)
            embedding_results.append(embedding_item)
        results = squeeze_same_results(embedding_results)
    if qubits is not None:
        new_results = []
        for item in results:
            new_results.append({
                'results': {key: item['results'][key] for key in qubits},
                'min_energy': item['min_energy'],
                'occurrences': item['occurrences']
            })
        results = new_results
    return squeeze_same_results(results)


def print_response(response, embedding=None, qubits=None):
    results = get_response(response, embedding=embedding, qubits=qubits)
    for item in results:
        print(item['results'], "Minimum energy: ", item['min_energy'],
              f"Occurrences: {item['occurrences']:.2f}%")


# TODO Need refactoring
def get_response_only_minimal(response, embedding=None, qubits=None):
    results = get_response(response, embedding=embedding, qubits=qubits)
    min_energy = min(map(lambda x: x['min_energy'], results))
    results = list(filter(lambda x: x['min_energy'] == min_energy, results))
    return squeeze_same_results(results)


def print_response_only_minimal(response, embedding=None, qubits=None):
    results_only_minimal = get_response_only_minimal(response, embedding=embedding,
                                                     qubits=qubits)
    for item in results_only_minimal:
        print(item['results'], "Minimum energy: ", item['min_energy'],
              f"Occurrences: {item['occurrences']:.2f}%")
    print()
    total = sum([item['occurrences'] for item in results_only_minimal])
    print(f"Total: {total:.2f}%")
              
def make_weight_bits_lin(containers, Wp, lambd): #linear part for square method
    weight_bits_lin = dict()
    for x in range(len(containers['Container ID'])):
        weight_bits_lin[x] = (-2*(containers.iloc[x]['Container mass (kg)'])/Wp + ((containers.iloc[x]['Container mass (kg)'])/Wp)**2)*lambd
    return weight_bits_lin

def make_weight_bits_sq_ij(containers, Wp, lambd): #quad part for square method
    weight_bits_sq_ij = dict()
    for x in range(len(containers['Container ID'])):
        for y in range(x + 1,len(containers['Container ID'])):
            weight_bits_sq_ij[(x,y)] = (2*((containers.iloc[x]['Container mass (kg)'])/Wp)*((containers.iloc[y]['Container mass (kg)'])/Wp))*lambd
    return weight_bits_sq_ij

def make_bqm_weight_sq(containers, Wp, bias, lambd): #square method
    weight_bits_lin = make_weight_bits_lin(containers, Wp, lambd)
    weight_bits_sq_ij = make_weight_bits_sq_ij(containers, Wp, lambd)
    weights_bqm =  dimod.BinaryQuadraticModel(weight_bits_lin, weight_bits_sq_ij, bias, dimod.BINARY)
    return weights_bqm

def make_bqm_weight_lin(containers, Wp, bias, lambd):
    weight_bits_lin = dict()
    for x in range(len(containers['Container ID'])):
        weight_bits_lin[x] = -(containers.iloc[x]['Container mass (kg)'])/Wp*lambd
    weight_bqm =  dimod.BinaryQuadraticModel(weight_bits_lin, {}, bias, dimod.BINARY)
    return weight_bqm

def make_val_bits_lin_in_sq(containers, L, lambd): #linear part for square method
    val_bits_lin = dict()
    for x in range(len(containers['Container ID'])):
        val_bits_lin[x] = (-2*(containers.iloc[x]['val'])/L + ((containers.iloc[x]['val'])/L)**2)*lambd
    return val_bits_lin

def make_val_bits_sq_ij(containers, L, lambd): #quad part for square method
    val_bits_sq_ij = dict()
    for x in range(len(containers['Container ID'])):
        for y in range(x + 1,len(containers['Container ID'])):
            val_bits_sq_ij[(x,y)] = (2*((containers.iloc[x]['val'])/L)*((containers.iloc[y]['val'])/L))*lambd
    return val_bits_sq_ij

def make_bqm_val_sq(containers, L, bias, lambd): # square method
    val_bits_lin = make_val_bits_lin_in_sq(containers, L, lambd)
    val_bits_sq_ij = make_val_bits_sq_ij(containers, L, lambd)
    val_bqm =  dimod.BinaryQuadraticModel(val_bits_lin, val_bits_sq_ij, bias, dimod.BINARY)
    return val_bqm

def make_bqm_val_lin(containers, L, bias, lambd): #linear method
    val_bits_lin = dict()
    for x in range(len(containers['Container ID'])):
        val_bits_lin[x] = lambd*(containers.iloc[x]['val'])/L
    val_bqm =  dimod.BinaryQuadraticModel(val_bits_lin, {}, bias, dimod.BINARY)
    return val_bqm

def run(bqm, num_reads):
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads)
    return get_response(response)

def sum_of_bqm(bqm1, bqm2):
    
    bqm1.add_variables_from(bqm2.linear)
    bqm1.add_interactions_from(bqm2.quadratic)
    bqm1.add_offset(bqm2.offset)
    return bqm1

def make_containers_with_bits(containers, results, num_res_with_min_energy, start): #results after def run
    for x in range(start, start + num_res_with_min_energy):
        res = pd.DataFrame.from_dict(results[x - start]['results'], orient='index')
        containers[x] = res[0]
    return containers

def calculate_weight_for_res(containers, num_res_with_min_energy, exp_number): #containers with bits
    sum_weight_in_min_energy = dict()
    for x in range(num_res_with_min_energy):
        bits = np.array(containers[x + exp_number])
        mass = np.array(containers['Container mass (kg)'])
        sum_weight_in_min_energy[x]= bits@mass.transpose()
    return sum_weight_in_min_energy

def calculate_val_for_res(containers, num_res_with_min_energy, exp_number): #containers with bits
    sum_val_in_min_energy = dict()
    for x in range(num_res_with_min_energy):
        bits = np.array(containers[x + exp_number])
        val = np.array(containers['val'])
        sum_val_in_min_energy[x]= bits@val.transpose()
    return sum_val_in_min_energy

def make_distr(containers, Wp, L, bias, lambd_val, lambd_weight, number_of_exp, num_res_with_min_energy, num_reads):
    bqm_weight = make_bqm_weight_sq(containers, Wp, bias, lambd_weight)
    bqm_val = make_bqm_val_sq(containers, L,bias, lambd_val)
    bqm = sum_of_bqm(bqm_weight, bqm_val)
    distr = dict()
    for count in range(number_of_exp):
        results = run(bqm, num_reads)
        containers = make_containers_with_bits(containers, results, num_res_with_min_energy, count)
        weight = calculate_weight_for_res(containers, num_res_with_min_energy, count)
        value = calculate_val_for_res(containers, num_res_with_min_energy, count)
        distr[count] = (value[0],weight[0])
    return distr

def make_distr_w(containers, Wp, L, bias, lambd_w, number_of_exp, num_res_with_min_energy, num_reads):
    bqm_weight = make_bqm_weight_sq(containers, Wp, bias, lambd_w)
    distr = dict()
    for count in range(number_of_exp):
        results = run(bqm_weight, num_reads)
        containers = make_containers_with_bits(containers, results, num_res_with_min_energy, count)
        weight = calculate_weight_for_res(containers, num_res_with_min_energy, count)
        value = calculate_val_for_res(containers, num_res_with_min_energy, count)
        distr[count] = (value[0],weight[0])
    return distr