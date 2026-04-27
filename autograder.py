"""
Autograder for Homework 4 [Spring 2026]

The autograder evaluates students implementation of normalization, dependency testing
and vectorization against a set of 25 test cases. The maximum points achievable on the
assignment is 25 (i.e. 1 point per test case). The test cases are split as follows
    1. Normalization Test [2 points]
    2. Dependency Testing [8 points]
    3. Unvectorizable cases [3 points]
    4. Vectorizable cases [10 points]
        - 0.5 points given if one of dependency testing is correct
    5. Vectorization with normalization and constant folding [2 points]
        - 0.5 points given if normalization is correct
"""

import argparse
import numpy as np
import csv
import re
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from autovec.codegen import NumpyBuffer
from autovec.simple_lang.compiler import SimpleLang2CCompiler
from autovec.simple_lang.vectorizer.dependency_graph import (
    construct_dependency_graph,
    DependencyGraphNode,
)
from autovec.simple_lang.vectorizer.normalize import normalize
from autovec.simple_lang.vectorizer.vectorize import vectorize
from autovec.simple_lang.vectorizer.dependency_testing import dependency_test
from autovec.simple_lang.parser import SimpleLangParser
import autovec.simple_lang.nodes as smpl
import copy

# Test Case Format: [Input, Normalized Output, Dependency Graph, Vectorized Output, [Numpy Input, Numpy Output]]
# In certain cases certain fields are marked None, if they are not relevant or they do not contribute to partial credit
# In cases of ZIV two possible dependency graphs are listed, a match with either grants student full points.
tests = {
    # Normalize: General
    "normalize1": [
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,8,2)
                A[i] = 0
            end
            return A
        end
        """,
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,4,1)
                A[2*i] = 0
            end
            return A
        end
        """,
        None,
        None,
        None,
    ],
    # Normalize: 2D
    "normalize2": [
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,8,2)
                for j in range(0,8,4)
                    A[i,j] = A[i+1,15] + 12
                end
            end
            return A
        end
        """,
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,4,1)
                for j in range(0,2,1)
                    A[2*i,4*j] = A[2*i+1,15] + 12
                end
            end
            return A
        end
        """,
        None,
        None,
        None,
    ],
    # Strong SIV: General
    "dependency_graph_strong_siv1": [
        """
        function prgm(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
            for i in range(0,15,1)
                D[i] = A[i] + 4     # S1
                A[i+1] = B[i] + 6   # S2
                Y[i] = X[i] + D[i]  # S3
                X[i+1] = Y[i] + 9   # S4
            end
            return X
        end
        """,
        None,
        (
            # fmt: off
            { "S1": [("S3", "i")], "S2": [("S1", "i")], "S3": [("S4", "i")], "S4": [("S3", "i")],},
            # fmt: on
        ),
        None,
        None,
    ],
    # Strong SIV: Tests if bounds if distance is between upper and lower bound
    "dependency_graph_strong_siv2": [
        """
        function prgm(A[64]) -> [64]:
            for i in range(0,16,1)
                A[i+32] = A[i+16] + 12  # S1
            end
            return A
        end
        """,
        None,
        ({"S1": []},),
        None,
        None,
    ],
    # Strong SIV: Tests if bounds if distance is integer
    "dependency_graph_strong_siv3": [
        """
        function prgm(A[64]) -> [64]:
            for i in range(0,16,1)
                A[2*i] = A[2*i+1] + 12  # S1
            end
            return A
        end
        """,
        None,
        ({"S1": []},),
        None,
        None,
    ],
    # Weak Zero SIV: General
    "dependency_graph_weak_zero_siv1": [
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(0,16,1)
                A[i] = 1          # S1
                B[i] = A[5] + 2   # S2
            end
            return B
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": [("S1", "i")]},),
        None,
        None,
    ],
    # Weak Zero SIV: 2D and self-loop
    "dependency_graph_weak_zero_siv2": [
        """
        function prgm(A[16,16]) -> [16]:
            for i in range(0,16,1)
                A[i,15] = A[0,15] + A[15,15]     # S1
            end
            return A
        end
        """,
        None,
        ({"S1": [("S1", "i")]},),
        None,
        None,
    ],
    # Weak Zero SIV: Integer value check
    "dependency_graph_weak_zero_siv3": [
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,8,1)
                A[2*i] = A[3] + A[15]     # S1
            end
            return A
        end
        """,
        None,
        ({"S1": []}),
        None,
        None,
    ],
    # ZIV: General
    "dependency_graph_ziv1": [
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(0,16,1)
                A[0] = 1          # S1
                B[i] = A[0] + 2   # S2
            end
            return B
        end
        """,
        None,
        (
            {"S1": [("S1", "ziv"), ("S2", "ziv")], "S2": [("S1", "ziv")]},
            {"S1": [("S1", "i"), ("S2", "i")], "S2": [("S1", "i")]},
        ),
        None,
        None,
    ],
    # ZIV: Negative
    "dependency_graph_ziv2": [
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(0,16,1)
                A[0] = 1          # S1
                B[i] = A[1] + 2   # S2
            end
            return B
        end
        """,
        None,
        ({"S1": [("S1", "ziv")], "S2": []}, {"S1": [("S1", "i")], "S2": []}),
        None,
        None,
    ],
    # Unvectorizable: Strong SIV
    "unvectorizable1": [
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,15,1)
                A[i+1] = A[i] + 12  # S1
            end
            return A
        end
        """,
        None,
        ({"S1": [("S1", "i")]},),
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,15,1)
                A[i+1] = A[i] + 12
            end
            return A
        end
        """,
        (
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133, 145, 157, 169, 181],
                dtype=np.float64,
            ),
        ),
    ],
    # Unvectorizable: Strong SIV 2
    "unvectorizable2": [
        """
        function prgm(A[16], F[16]) -> [16]:
            for i in range(0,15,1)
                A[i+1] = F[i]   # S1
                F[i+1] = A[i]   # S2
            end
            return F
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": [("S1", "i")]},),
        """
        function prgm(A[16], F[16]) -> [16]:
            for i in range(0,15,1)
                A[i+1] = F[i]
                F[i+1] = A[i]
            end
            return F
        end
        """,
        (
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.float64,
            ),
        ),
    ],
    # Unvectorizable: Weak SIV
    "unvectorizable3": [
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(0,16,1)
                A[i] = 1          # S1
                B[i] = A[5] + 2   # S2
            end
            return B
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": [("S1", "i")]},),
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(0,16,1)
                A[i] = 1
                B[i] = A[5] + 2
            end
            return B
        end
        """,
        (
            (
                np.array([i for i in range(0, 16)], dtype=np.float64),
                np.zeros(16, dtype=np.float64),
            ),
            np.array([7 if i < 5 else 3 for i in range(0, 16)], dtype=np.float64),
        ),
    ],
    # Vectorize: General
    "vectorizable1": [
        """
        function prgm(A[16]) -> [16]:
            for i in range(0,16,1)
                A[i] = 0    # S1
            end
            return A
        end
        """,
        None,
        ({"S1": []},),
        """
        function prgm(A[16]) -> [16]:
            A[0:16] = 0
            return A
        end
        """,
        (
            (np.full(shape=(16,), fill_value=5, dtype=np.float64),),
            np.full(shape=(16,), fill_value=0, dtype=np.float64),
        ),
    ],
    # Vectorize: Loop independent
    "vectorizable2": [
        """
        function prgm(A[16], B[16], C[16]) -> [16]:
            for i in range(0,16,1)
                A[i] = B[i]     # S1
                C[i] = A[i]     # S2
            end
            return C
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": []},),
        """
        function prgm(A[16], B[16], C[16]) -> [16]:
            A[0:16] = B[0:16]
            C[0:16] = A[0:16]
            return C
        end
        """,
        (
            (
                np.full(shape=(16,), fill_value=0, dtype=np.float64),
                np.full(shape=(16,), fill_value=7, dtype=np.float64),
                np.full(shape=(16,), fill_value=0, dtype=np.float64),
            ),
            np.full(shape=(16,), fill_value=7, dtype=np.float64),
        ),
    ],
    # Vectorize: Loop Carried
    "loop_splitting1": [
        """
        function prgm(A[16], B[16]) -> [16]:
            for i in range(1,16,1)
                A[i] = B[i] + 1     # S1
                B[i-1] = A[i] - 5   # S2
            end
            return B
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": []},),
        """
        function prgm(A[16], B[16]) -> [16]:
            A[1:16] = B[1:16] + 1
            B[0:15] = A[1:16] - 5
            return B
        end
        """,
        (
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
            ),
            np.array(
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: Loop Carried 2
    "loop_splitting2": [
        """
        function prgm(A[16], B[16], D[16]) -> [16]:
            for i in range(0,15,1)
                A[i+1] = B[i] + 1   # S1
                D[i] = A[i] - 5     # S2
            end
            return D
        end
        """,
        None,
        ({"S1": [("S2", "i")], "S2": []},),
        """
        function prgm(A[16], B[16], D[16]) -> [16]:
            A[1:16] = B[0:15] + 1
            D[0:15] = A[0:15] - 5
            return D
        end
        """,
        (
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.full(shape=(16,), fill_value=0, dtype=np.float64),
            ),
            np.array(
                [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: Simple Vectorization 1
    "simple_vectorization_algo1": [
        """
        function prgm(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
            for i in range(0,15,1)
                D[i] = A[i] + 4     # S1
                A[i+1] = B[i] + 6   # S2
                Y[i] = X[i] + D[i]  # S3
                X[i+1] = Y[i] + 9   # S4
            end
            return X
        end
        """,
        None,
        (
            {
                "S1": [("S3", "i")],
                "S2": [("S1", "i")],
                "S3": [("S4", "i")],
                "S4": [("S3", "i")],
            },
        ),
        """
        function prgm(A[16], B[16], D[16], X[16], Y[16]) -> [16]:
            A[1:16] = B[0:15] + 6
            D[0:15] = A[0:15] + 4
            for i in range(0,15,1)
                Y[i] = X[i] + D[i]
                X[i+1] = Y[i] + 9
            end
            return X
        end
        """,
        (
            (
                np.full(shape=(16,), fill_value=1, dtype=np.float64),
                np.full(shape=(16,), fill_value=2, dtype=np.float64),
                np.full(shape=(16,), fill_value=3, dtype=np.float64),
                np.full(shape=(16,), fill_value=4, dtype=np.float64),
                np.full(shape=(16,), fill_value=5, dtype=np.float64),
            ),
            # fmt: off
            np.array([4, 18, 39, 60, 81, 102, 123, 144, 165, 186, 207, 228, 249, 270, 291, 312], dtype=np.float64),
            # fmt: on
        ),
    ],
    # Vectorize: Simple Vectorization 2
    "simple_vectorization_algo2": [
        """
        function prgm(A[16], B[16], D[16]) -> [16]:
            for i in range(1,16,1)
                D[i] = A[i-1] + 1   # S1
                A[i] = B[i]         # S2
            end
            return D
        end
        """,
        None,
        ({"S1": [], "S2": [("S1", "i")]},),
        """
        function prgm(A[16], B[16], D[16]) -> [16]:
            A[1:16] = B[1:16]
            D[1:16] = A[0:15] + 1
            return D
        end
        """,
        (
            (
                np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    dtype=np.float64,
                ),
                np.array(
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                    dtype=np.float64,
                ),
                np.full(shape=(16,), fill_value=0, dtype=np.float64),
            ),
            np.array(
                [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: Advanced Vectorization 1
    "adv_vectorization_algo1": [
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,15,1)
                for j in range(0,16,1)
                    A[i+1,j] = A[i,j] + 1   # S1
                end
            end
            return A
        end
        """,
        None,
        ({"S1": [("S1", "i")]},),
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,15,1)
                A[i+1,0:16] = A[i,0:16] + 1
            end
            return A
        end
        """,
        (
            (np.full(shape=(16, 16), fill_value=0, dtype=np.float64),),
            np.array([[i] * 16 for i in range(0, 16)], dtype=np.float64),
        ),
    ],
    # Vectorize: Advanced Vectorization 2
    "adv_vectorization_algo2": [
        """
        function prgm(A[16,16,16]) -> [16,16,16]:
            for i in range(0,16,1)
                for j in range(0,15,1)
                    for k in range(0,16,1)
                        A[i,j+1,k] = A[i,j,k] + 1   # S1
                    end
                end
            end
            return A
        end
        """,
        None,
        ({"S1": [("S1", "j")]},),
        """
        function prgm(A[16,16,16]) -> [16,16,16]:
            for i in range(0,16,1)
                for j in range(0,15,1)
                    A[i,j+1,0:16] = A[i,j,0:16] + 1
                end
            end
            return A
        end
        """,
        (
            (np.full(shape=(16, 16, 16), fill_value=0, dtype=np.float64),),
            np.array(
                [[[i] * 16 for i in range(0, 16)] for j in range(0, 16)],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: Weak SIV 1
    "weak_siv1": [
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,16,1)
                for j in range(0,16,1)
                    A[i,j] = A[5,j] + 2     # S1
                end
            end
            return A
        end
        """,
        None,
        ({"S1": [("S1", "i")]},),
        """
        function prgm(A[16,16]) -> [16,16]:
            for i in range(0,16,1)
                A[i,0:16] = A[5,0:16] + 2
            end
            return A
        end
        """,
        (
            (np.array([[i] * 16 for i in range(0, 16)], dtype=np.float64),),
            np.array(
                [[7 if i <= 5 else 9] * 16 for i in range(0, 16)], dtype=np.float64
            ),
        ),
    ],
    # Vectorize: Multi-Loop Case
    "multi_loop": [
        """
        function prgm(A[16,16], B[16], C[16]) -> [16,16]:
            for k in range(0,16,1)
                B[k] = C[k]
            end

            for i in range(0,15,1)
                for j in range(0,16,1)
                    A[i+1,j] = A[i,j] + B[i]
                end
            end
            return A
        end
        """,
        None,
        None,
        """
        function prgm(A[16,16], B[16], C[16]) -> [16,16]:
            B[0:16] = C[0:16]
            for i in range(0,15,1)
                A[i+1,0:16] = A[i,0:16] + B[i]
            end
            return A
        end
        """,
        (
            (
                np.array([[i] * 16 for i in range(0, 16)], dtype=np.float64),
                np.zeros(16, dtype=np.float64),
                np.array([i for i in range(0, 16)], dtype=np.float64),
            ),
            np.array(
                [
                    [0.0] * 16,
                    [0.0] * 16,
                    [1.0] * 16,
                    [3.0] * 16,
                    [6.0] * 16,
                    [10.0] * 16,
                    [15.0] * 16,
                    [21.0] * 16,
                    [28.0] * 16,
                    [36.0] * 16,
                    [45.0] * 16,
                    [55.0] * 16,
                    [66.0] * 16,
                    [78.0] * 16,
                    [91.0] * 16,
                    [105.0] * 16,
                ],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: with normalize
    "vectorize_normalize": [
        """
        function prgm(A[16,32]) -> [16,32]:
            for i in range(0,15,1)
                for j in range(16,31,1)
                    A[i+1,j+1] = A[i,11] + 2
                end
            end
            return A
        end
        """,
        """
        function prgm(A[16,32]) -> [16,32]:
            for i in range(0,15,1)
                for j in range(0,15,1)
                    A[i+1,j+17] = A[i,11] + 2
                end
            end
            return A
        end
        """,
        None,
        """
        function prgm(A[16,32]) -> [16,32]:
            for i in range(0,15,1)
                A[i+1,17:32] = A[i,11] + 2
            end
            return A
        end
        """,
        (
            (
                np.array(
                    [[i for i in range(0, 32)] for j in range(0, 16)], dtype=np.float64
                ),
            ),
            np.array(
                [
                    (
                        [i for i in range(0, 32)]
                        if j == 0
                        else [i if i < 17 else 13 for i in range(0, 32)]
                    )
                    for j in range(0, 16)
                ],
                dtype=np.float64,
            ),
        ),
    ],
    # Vectorize: Constant Folding
    "vectorize_cf": [
        """
        function prgm(A[32]) -> [32]:
            for i in range(0,16,1)
                A[i] = A[i+4*4] + 12  # S1
            end
            return A
        end
        """,
        None,
        ({"S1": []},),
        """
        function prgm(A[32]) -> [32]:
            A[0:16] = A[16:32] + 12
            return A
        end
        """,
        (
            (np.array([i for i in range(0, 32)], dtype=np.float64),),
            np.array([i + 28 if i < 16 else i for i in range(0, 32)], dtype=np.float64),
        ),
    ],
}


def test_dependency_graph(input_prgm, expected_dependency_graph):
    parser = SimpleLangParser()
    output_prgm = parser.parse(input_prgm)

    # resetting the count to make verification easy
    DependencyGraphNode.count = 0

    # We assume there is only one loop per test case
    dependency_graph = construct_dependency_graph(
        output_prgm.body.bodies[0], dependency_test
    )

    # Simplify dependency graph for verification
    simplified_dependency_graph: dict[str, list[str]] = {}
    for src, children in dependency_graph.items():
        simplified_dependency_graph[src.unique_id] = sorted(
            [(child.target.unique_id, child.loop_lvl.name) for child in children]
        )
    return any(simplified_dependency_graph == g for g in expected_dependency_graph)


def test_normalize(input_prgm, expected_output_prgm):
    parsed_prgm = SimpleLangParser().parse(input_prgm)
    normalized_prgm = normalize(parsed_prgm.body.bodies[0])
    return (
        normalized_prgm == SimpleLangParser().parse(expected_output_prgm).body.bodies[0]
    )


def get_variables(node):
    match node:
        case smpl.Variable(name, type):
            return {name}
        case smpl.Load(buf, idxs):
            return {buf.name}
        case smpl.Call(_, args):
            var_names = set()
            for arg in args:
                var_names.update(get_variables(arg))
            return var_names
    return set()


def dce_block(block, used_vars):
    new_bodies = []
    for stmt in reversed(block.bodies):
        match stmt:
            case smpl.Return(arg):
                used_vars.update(get_variables(arg))
                new_bodies.append(stmt)
            case smpl.Store(buf, idxs, val):
                if buf.name in used_vars:
                    used_vars.update(get_variables(val))
                    new_bodies.append(stmt)
            case smpl.ForLoop(lvl, start, end, stride, body):
                # Repeat till we reach fixed point for dataflow
                prev_used_vars_set = set()
                first_pass = True
                while first_pass or prev_used_vars_set != used_vars:
                    first_pass = False
                    prev_used_vars_set = copy.deepcopy(used_vars)
                    new_body = dce_block(body, used_vars)

                if len(new_body.bodies) > 0:
                    new_loop = smpl.ForLoop(lvl, start, end, stride, new_body)
                    new_bodies.append(new_loop)
            case _:
                raise Exception(f"Unrecognized node {stmt}")

    return smpl.Block(tuple(reversed(new_bodies)))


def dce(func_node):
    used_vars = set()
    new_body = dce_block(func_node.body, used_vars)
    return smpl.Function(func_node.name, func_node.args, new_body)


def has_vector_index(node):
    match node:
        case smpl.Function(_, _, body):
            return has_vector_index(body)
        case smpl.Block(bodies):
            return any(has_vector_index(body) for body in bodies)
        case smpl.ForLoop(_, _, _, _, body):
            return has_vector_index(body)
        case smpl.Store(buf, idxs, val):
            return any(has_vector_index(idx) for idx in idxs) or has_vector_index(val)
        case smpl.Load(buf, idxs):
            return any(has_vector_index(idx) for idx in idxs)
        case smpl.Call(_, args):
            return any(has_vector_index(arg) for arg in args)
        case smpl.VectorIndex(_, _, _):
            return True
        case _:
            return False
    return False


def test_execute(input_prgm, input, expected_out, is_vectorizable):
    compiler = SimpleLang2CCompiler()
    mod = compiler(input_prgm, 8)

    buf_input = [NumpyBuffer(arr) for arr in input]
    result = mod.prgm(*buf_input)
    if not np.allclose(result.arr, expected_out):
        return False

    parsed_prgm = SimpleLangParser().parse(input_prgm)
    vectorized_ast = vectorize(parsed_prgm, dependency_test)

    new_func = dce(vectorized_ast)

    has_vec = has_vector_index(new_func)

    if is_vectorizable:
        return has_vec
    else:
        return not has_vec


class Stage(Enum):
    failed = 0
    normalize = 1
    dependency_graph = 2
    execute = 3


class TestResult:
    def __init__(self, test_case_name, passed, stageOfLastPass):
        self.test_case_name = test_case_name
        self.passed = passed
        self.stageOfLastPass = stageOfLastPass


class ScoreManager:
    def __init__(self):
        self.results = []

    def get_fully_passed_functions(self):
        return [r.test_case_name for r in self.results if r.passed]

    def get_failed_functions(self):
        return [
            f"{r.test_case_name}({r.stageOfLastPass.name})"
            for r in self.results
            if not r.passed
        ]

    def get_final_score(self):
        score = 0.0
        for r in self.results:
            # We don't hardcode execute here since some tests may exist
            # only for certain tests
            if r.passed:
                score += 1.0
            elif r.stageOfLastPass == Stage.dependency_graph:
                score += 0.50
            elif r.stageOfLastPass == Stage.normalize:
                score += 0.50
        return score

    def record_result(self, result):
        self.results.append(result)


def test_dce():
    unvectorized_prgm = """
    function prgm(A[16,16]) -> [16,16]:
        for i in range(0,15,1)
            for j in range(0,16,1)
                A[i+1,j] = A[i,j] + 1   # S1
            end
        end
        return A
    end
    """

    vectorized_prgm = """
    function prgm(A[16,16]) -> [16,16]:
        for i in range(0,15,1)
            A[i+1,0:16] = A[i,0:16] + 1
        end
        return A
    end
    """

    vectorized_prgm_with_dead_code = """
    function prgm(A[16,16], B[16,16]) -> [16,16]:
        for i in range(0,15,1)
            for j in range(0,16,1)
                A[i+1,j] = A[i,j] + 1   # S1
                B[0:16,j] = B[0:16,j] + 1
            end
        end
        return A
    end
    """

    parsed_prgm = SimpleLangParser().parse(unvectorized_prgm)
    prgm_with_dce = dce(parsed_prgm)
    assert has_vector_index(prgm_with_dce) == False

    parsed_prgm = SimpleLangParser().parse(vectorized_prgm)
    prgm_with_dce = dce(parsed_prgm)
    assert has_vector_index(prgm_with_dce) == True

    parsed_prgm = SimpleLangParser().parse(vectorized_prgm_with_dead_code)
    prgm_with_dce = dce(parsed_prgm)
    assert has_vector_index(prgm_with_dce) == False

    print("DCE passed test cases!")


def run_single_test(test_name, test_data):
    """Run a single test and return the result."""
    input_prgm = test_data[0]
    normalized_prgm = test_data[1]
    expected_dep = test_data[2]
    expected_vec = test_data[3]

    last_passed_stage = Stage.failed
    passed = False

    is_vectorizable = re.match(r"^unvectorizable.*", test_name) is None

    if expected_dep is None and expected_vec is None:
        # Only normalization case
        try:
            if test_normalize(input_prgm, normalized_prgm):
                last_passed_stage = Stage.normalize
                passed = True
        except Exception:
            pass

    elif normalized_prgm is None and expected_vec is None:
        # Only dependency test case
        try:
            if test_dependency_graph(input_prgm, expected_dep):
                last_passed_stage = Stage.dependency_graph
                passed = True
        except Exception:
            pass

    else:
        # All case
        inputs = test_data[4][0]
        expected_out = test_data[4][1]

        try:
            if test_execute(input_prgm, inputs, expected_out, is_vectorizable):
                last_passed_stage = Stage.execute
                passed = True
        except Exception:
            pass

        if expected_dep is not None and last_passed_stage == Stage.failed:
            try:
                if test_dependency_graph(input_prgm, expected_dep):
                    last_passed_stage = Stage.dependency_graph
            except Exception:
                pass

        # Only test normalization if that is provided
        if normalized_prgm is not None and last_passed_stage == Stage.failed:
            try:
                if test_normalize(input_prgm, normalized_prgm):
                    last_passed_stage = Stage.normalize
            except Exception:
                pass

    return TestResult(test_name, passed, last_passed_stage)


if __name__ == "__main__":
    """
    Usage: python autograder.py <student_name> <output_csv>
        student_name: Name of the student submitting
        output_csv: Path to CSV file for results (will be appended)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("student_name")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    sm = ScoreManager()

    for test_name, test_data in tests.items():
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_single_test, test_name, test_data)
                result = future.result(timeout=30)
                sm.record_result(result)
        except TimeoutError:
            # Test took more than 30 seconds, record as failed
            sm.record_result(TestResult(test_name, False, Stage.failed))
        except Exception as e:
            # Handle any other unexpected exceptions
            sm.record_result(TestResult(test_name, False, Stage.failed))

    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                args.student_name,
                ", ".join(sm.get_failed_functions()),
                ", ".join(sm.get_fully_passed_functions()),
                sm.get_final_score(),
            ]
        )
