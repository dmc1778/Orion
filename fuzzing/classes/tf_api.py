import re
from functools import WRAPPER_UPDATES
import inspect
import json
import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from numpy.random import choice, randint
import string
# from tensorflow.python.eager.remote import connect_to_cluster
from constants.keys import *
from classes.argument import ArgType, Argument
from classes.api import API
from termcolor import colored
from classes.rules import MakeNegative
from classes.api import API
from classes.database import TFDatabase

from classes.argument import OracleType
from utils.probability import do_type_mutation, do_select_from_db, change_dim


"""
Imports
"""

# ADDED by Nima
from tensorflow.python.autograph.pyct.static_analysis import activity
# from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.tpu.ops import tpu_ops
# from tensorflow.python.keras.benchmarks.saved_model_benchmarks import saved_model_benchmark_util
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops.linalg import solve_registrations
# from tensorflow.python.util import compat_internal
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.summary import summary_iterator
from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.keras.saving.saved_model import save
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import config
# from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import state_ops
from tensorflow.python.data.util import convert
from tensorflow.python.ops.signal import mfcc_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.util import lock_util
from tensorflow.python.ops.numpy_ops import np_utils
# from tensorflow.python.keras.distribute import keras_utils_test
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.saved_model import simple_save
from tensorflow.python.debug.cli import cli_test_utils
# from tensorflow.python.keras.distribute import keras_dnn_correctness_test
from tensorflow.python.framework import auto_control_deps_utils
# from tensorflow.python.keras.wrappers import scikit_learn
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.distribute.cluster_resolver import slurm_cluster_resolver
from tensorflow.python.training import summary_io
from tensorflow.python.saved_model import load_options
from tensorflow.python.ops import io_ops
# from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops.distributions import multinomial
from tensorflow.python.types import doc_typealias
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.training import evaluation
from tensorflow.python.keras.engine import base_layer
# from tensorflow.python.keras.applications import resnet_v2
from tensorflow.python.data.experimental.ops import parsing_ops
# from tensorflow.python.keras.applications import inception_v3
# from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops.linalg import linear_operator_toeplitz
from tensorflow.python.ops import numerics
from tensorflow.python.util import deprecation
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import tf_contextlib
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.layers import utils
from tensorflow.python.eager import tape
from tensorflow.python.training.tracking import tracking
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.eager import cancellation
# from tensorflow.python.keras.layers.preprocessing import text_vectorization
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.training import input
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.ops.linalg import linalg_impl
# from tensorflow.python.keras.datasets import boston_housing
from tensorflow.python.distribute import distribute_config
# from tensorflow.python.keras.distribute import simple_models
from tensorflow.python.ops import linalg_grad
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.keras.saving.saved_model import json_utils
# from tensorflow.python.distribute import all_reduce
from tensorflow.python.eager import core
from tensorflow.python.util import compat
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.ops.distributions import laplace
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops import gen_audio_ops
from tensorflow.python.ops.distributions import dirichlet_multinomial
# from tensorflow.python.keras.applications import vgg19
from tensorflow.python.ops import clustering_ops
# from tensorflow.python.keras.layers import cudnn_recurrent
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.feature_column import utils
# from tensorflow.python.keras.legacy_tf_layers import normalization
from tensorflow.python.util import lazy_loader
from tensorflow.python.framework import sparse_tensor
# from tensorflow.python.keras.layers.preprocessing import integer_lookup_v1
# from tensorflow.python.ops import control_flow_state
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.util import function_utils
from tensorflow.python.training import basic_loops
from tensorflow.python.ops.linalg import cholesky_registrations
from tensorflow.python.util import nest
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.framework import indexed_slices
# from tensorflow.python.keras.distribute import keras_rnn_model_correctness_test
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.keras.optimizer_v2 import utils
from tensorflow.python.eager import profiler
from tensorflow.python.distribute.cluster_resolver import cluster_resolver
# from tensorflow.python.keras.layers.preprocessing import category_encoding_v1
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.eager import forwardprop
from tensorflow.python.ops.linalg import linear_operator_low_rank_update
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.keras.mixed_precision import test_util
from tensorflow.python.training import checkpoint_management
# from tensorflow.python.keras.layers.preprocessing import hashing
from tensorflow.python.training import slot_creator
from tensorflow.python.distribute import input_lib
from tensorflow.python.debug.cli import curses_widgets
from tensorflow.python.util import tf_decorator
from tensorflow.python.eager import def_function
from tensorflow.python.framework import error_interpolation
from tensorflow.python.ops import gen_set_ops
from tensorflow.python.data.util import random_seed
# from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.ops.linalg import linear_operator_tridiag
# from tensorflow.python.keras.layers.preprocessing import preprocessing_stage
from tensorflow.python.keras.engine import functional
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.framework import meta_graph
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gen_encode_proto_ops
from tensorflow.python.eager import profiler_client
from tensorflow.python.training.experimental import loss_scale_optimizer
from tensorflow.python.autograph.operators import exceptions
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.autograph.utils import context_managers
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.keras.saving import save
from tensorflow.python.types import distribute
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.keras.initializers import initializers_v2
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.grappler import item
from tensorflow.python.ops.parallel_for import pfor
# from tensorflow.python.debug.examples.v1 import debug_mnist_v1
# from tensorflow.python.keras.distribute import distribute_strategy_test
from tensorflow.python.keras.engine import node
from tensorflow.python.autograph.operators import variables
# from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.feature_column import feature_column
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.framework import test_combinations
# from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.legacy_tf_layers import convolutional
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.framework import subscribe
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.util import tf_export
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.grappler import cluster
from tensorflow.python.platform import gfile
from tensorflow.python.autograph.converters import functions
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import c_api_util
from tensorflow.python.eager import function
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.keras.saving import model_config
from tensorflow.python.eager import execute
from tensorflow.python.keras.legacy_tf_layers import pooling
from tensorflow.python.keras.engine import base_layer_v1
from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.training import session_run_hook
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.keras import optimizers
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.data.experimental.ops import compression_ops
# from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.training import gradient_descent
from tensorflow.python.keras.saving.saved_model import save_impl
# from tensorflow.python.data.experimental.ops import stats_options
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.ops.distributions import transformed_distribution
from tensorflow.python.training.tracking import python_state
# from tensorflow.python.training.experimental import loss_scaling_gradient_tape
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.ops import session_ops
from tensorflow.python.util import tf_stack
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.util import all_util
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.keras.engine import sequential
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import rnn_grad
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.ops import gen_decode_proto_ops
from tensorflow.python.training.tracking import base
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.ops import check_ops
from tensorflow.python.keras.layers import core
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.util import example_parser_configuration
from tensorflow.python.ops.linalg import linear_operator_kronecker
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops.distributions import normal
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.distribute.cluster_resolver import kubernetes_cluster_resolver
from tensorflow.python.keras.utils import vis_utils
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.compiler.xla import jit
from tensorflow.python.ops.ragged import ragged_squeeze_op
from tensorflow.python.keras.engine import partial_batch_padding_handler
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.keras import activations
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import script_ops
# from tensorflow.python.keras.preprocessing import text_dataset
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.operators import slices
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.data.util import sparse
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import self_check
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.training import adadelta
from tensorflow.python.autograph.operators import logical
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.platform import googletest
from tensorflow.python.platform import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.distribute import collective_util
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.platform import analytics
from tensorflow.python.distribute import step_fn
# from tensorflow.python.keras.benchmarks import distribution_util
# from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.debug.cli import readline_ui
from tensorflow.python.distribute import estimator_training
from tensorflow.python.ops import collective_ops
from tensorflow.python.training import adam
from tensorflow.python.training import proximal_adagrad
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.platform import benchmark
from tensorflow.python.framework import load_library
from tensorflow.python.summary import summary
from tensorflow.python.keras import callbacks_v1
from tensorflow.python.autograph.pyct import gast_util
# from tensorflow.python.keras.applications import nasnet
# from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.ops.ragged import ragged_batch_gather_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.training.tracking import layer_utils
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.ops.distributions import exponential
from tensorflow.python.data.experimental.ops import io
# from tensorflow.python.keras.preprocessing import image_dataset
# from tensorflow.python.keras.layers.preprocessing import category_crossing
from tensorflow.python.ops.signal import reconstruction_ops
# from tensorflow.python.keras.layers import noise
from tensorflow.python.ops.ragged import ragged_operators
from tensorflow.python.training import momentum
from tensorflow.python.autograph.utils import testing
from tensorflow.python.keras.legacy_tf_layers import core
from tensorflow.python.ops import image_ops
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.data.util import structure
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.utils import tensor_list
# from tensorflow.python.saved_model import load_context
from tensorflow.python.ops import rnn
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.debug.cli import offline_analyzer
from tensorflow.python.keras.utils import losses_utils
# from tensorflow.python.keras.layers.preprocessing import image_preprocessing
# from tensorflow.python.keras.preprocessing import timeseries
from tensorflow.python.debug.cli import ui_factory
# from tensorflow.python.debug.examples.v2 import debug_fibonacci_v2
from tensorflow.python.keras.layers import serialization
from tensorflow.python.framework import combinations
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.autograph.pyct.common_transformers import anf
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.platform import tf_logging
from tensorflow.python.data.experimental.ops import cardinality
# from tensorflow.python.keras.distribute import keras_stateful_lstm_model_correctness_test
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import gen_nn_ops
# from tensorflow.python.keras.layers.preprocessing import string_lookup_v1
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.saved_model import main_op_impl
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.ops.ragged import ragged_concat_ops
# from tensorflow.python.keras.datasets import cifar
from tensorflow.python.distribute import values
from tensorflow.python.distribute import single_loss_example
from tensorflow.python.data.experimental.ops import unique
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_impl
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.ops.parallel_for import test_util
from tensorflow.python.distribute.cluster_resolver import gce_cluster_resolver
# from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.data.ops import dataset_ops
# from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.ops.distributions import util
from tensorflow.python.training import optimizer
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.ops import gen_control_flow_ops
# from tensorflow.python.ops import tensor_forest_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import resource_loader
from tensorflow.python.keras.mixed_precision import get_layer_policy
from tensorflow.python.autograph.converters import logical_expressions
# from tensorflow.python.keras.layers import wrappers
from tensorflow.python.framework import composite_tensor
from tensorflow.python.training import saver_test_utils
# from tensorflow.python.training import quantize_training
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.training import coordinator
from tensorflow.python.types import internal
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.training import server_lib
from tensorflow.python.framework import tensor_util
# from tensorflow.python.training import tensorboard_logging
from tensorflow.python.framework import common_shapes
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.util import decorator_utils
# from tensorflow.python.keras.datasets import imdb
from tensorflow.python.ops import op_selector
from tensorflow.python.ops.signal import shape_ops
# from tensorflow.python.keras.premade import wide_deep
from tensorflow.python.data.util import traverse
from tensorflow.python.autograph.converters import variables
from tensorflow.python.keras.layers import rnn_cell_wrapper_v2
from tensorflow.python.ops.linalg.sparse import conjugate_gradient
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.python.ops import gen_nccl_ops
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.training import rmsprop
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.distribute import packed_distributed_variable
from tensorflow.python.autograph.core import config_lib
# from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.ops import nn_grad
from tensorflow.python.keras.saving.saved_model import load
from tensorflow.python.util import object_identity
# from tensorflow.python.keras.tests import model_subclassing_test_util
from tensorflow.python.util.protobuf import compare
from tensorflow.python.data.experimental.ops import testing
# from tensorflow.python.keras.datasets import mnist
from tensorflow.python.ops import cond_v2
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.client import timeline
from tensorflow.python.keras import combinations
from tensorflow.python.ops import manip_grad
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.data.util import options
from tensorflow.python.data.experimental.ops import writers
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.util import serialization
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.keras.initializers import initializers_v1
from tensorflow.python.ops.linalg import linear_operator_full_matrix
# from tensorflow.python.keras.applications import xception
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_map_ops
# from tensorflow.python.keras.benchmarks import benchmark_util
# from tensorflow.python.keras.feature_column import dense_features_v2
from tensorflow.python.ops.distributions import dirichlet
from tensorflow.python.ops.linalg import linear_operator_permutation
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.ops import tensor_array_grad
# from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python.framework import test_ops
from tensorflow.python.data.ops import optional_ops
# from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.platform import sysconfig
from tensorflow.python.eager import context
from tensorflow.python.keras.engine import training
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops.signal import spectral_ops
from tensorflow.python.training.experimental import loss_scale
from tensorflow.python.client import device_lib
from tensorflow.python.ops import sparse_grad
from tensorflow.python.ops import special_math_ops
# from tensorflow.python.platform import status_bar
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.distribute import tpu_values
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.summary import plugin_asset
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.distribute.cluster_resolver import sagemaker_cluster_resolver
# from tensorflow.python.autograph.converters import control_flow_deprecated_py2
from tensorflow.python.distribute import ps_values
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import unconnected_gradients
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.eager import imperative_grad
from tensorflow.python.platform import remote_utils
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.training import device_setter
# from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.ops import bincount_ops
# from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.distribute import strategy_test_lib
# from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.ops.signal import mel_ops
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.ops.distributions import uniform
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.keras.optimizer_v2 import gradient_descent
# from tensorflow.python.debug.examples.v1 import debug_keras
from tensorflow.python.distribute import summary_op_util
# from tensorflow.python.autograph.operators import control_flow_deprecated_py2
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.training import session_manager
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.util import tf_inspect
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular
# from tensorflow.python.debug.examples.v1 import debug_fibonacci
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.signal import util_ops
# from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.autograph.operators import conditional_expressions
# from tensorflow.python.autograph.utils import py_func
from tensorflow.python.training import monitored_session
from tensorflow.python.distribute import device_util
from tensorflow.python.profiler.internal import flops_registry
from tensorflow.python.ops.linalg import adjoint_registrations
# from tensorflow.python.ops import gen_user_ops
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.eager import graph_only_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.distribute import reduce_util
from tensorflow.python.keras.legacy_tf_layers import base
from tensorflow.python.keras.utils import kernelized_utils
# from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.tpu.client import client
from tensorflow.python.ops import gradient_checker
from tensorflow.python.data.ops import multi_device_iterator_ops
# from tensorflow.python.keras.layers.preprocessing import reduction
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.ops import gen_manip_ops
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.framework import kernels
from tensorflow.python.ops import gen_boosted_trees_ops
# from tensorflow.python.keras.layers.preprocessing import normalization_v1
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.ops.linalg import inverse_registrations
from tensorflow.python.ops import array_grad
from tensorflow.python.util import tf_should_use
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.distribute import combinations
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.ops import functional_ops
# from tensorflow.python.saved_model import save_context
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.optimizer_v2 import legacy_learning_rate_decay
from tensorflow.python.ops.linalg import linear_operator_diag
# from tensorflow.python.autograph.utils import compat_util
# from tensorflow.python.keras.utils import multi_gpu_utils
from tensorflow.python.framework import traceable_stack
from tensorflow.python.keras.optimizer_v2 import adadelta
from tensorflow.python.keras import constraints
from tensorflow.python.ops import optional_grad
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras import models
# from tensorflow.python.keras.layers.ops import core
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.framework import func_graph
from tensorflow.python.distribute import values_util
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_script_ops
# from tensorflow.python.keras.tests import model_architectures
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.keras.layers import convolutional_recurrent
from tensorflow.python.framework import graph_util_impl
from tensorflow.python.ops import gradients_util
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import errors_impl
# from tensorflow.python.keras.applications import vgg16
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import sets_impl
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops.ragged import ragged_batch_gather_with_default_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.distribute import sharded_variable
# from tensorflow.python.keras.distribute import model_collection_base
# from tensorflow.python.framework import composite_tensor_utils
from tensorflow.python.keras import testing_utils
from tensorflow.python.data.experimental.ops import batching
# from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.module import module
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.keras import callbacks
from tensorflow.python.util import keyword_args
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops import variable_scope
from tensorflow.python.compiler.tensorrt import trt_convert
# from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.optimizer_v2 import adamax
from tensorflow.python.saved_model import method_name_updater
from tensorflow.python.platform import test
from tensorflow.python.eager import executor
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.ops import random_grad
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.ops.linalg import linear_operator_block_diag
# from tensorflow.python.keras.applications import mobilenet_v3
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.client import pywrap_tf_session
# from tensorflow.python.keras.distribute import keras_image_model_correctness_test
from tensorflow.python.keras.mixed_precision import loss_scale
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.training import supervisor
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.feature_column import serialization
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.optimizer_v2 import nadam
from tensorflow.python.training import warm_starting_util
from tensorflow.python.ops import gen_collective_ops
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_ragged_conversion_ops
# from tensorflow.python.keras.distribute import keras_embedding_model_correctness_test
from tensorflow.python.autograph.core import converter_testing
# from tensorflow.python.keras.engine import base_preprocessing_layer_v1
from tensorflow.python.keras import backend
from tensorflow.python.ops.distributions import categorical
# from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.framework import ops
# from tensorflow.python.keras.layers import normalization_v2
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.debug.cli import curses_ui
from tensorflow.python.training import ftrl
from tensorflow.python.saved_model import revived_types
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.util import dispatch
from tensorflow.python.ops.linalg import registrations_util
from tensorflow.python.ops import init_ops
from tensorflow.python.autograph.lang import directives
from tensorflow.python.keras import keras_parameterized
# from tensorflow.python.keras.applications import efficientnet
from tensorflow.python.ops import resources
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.ops import manip_ops
from tensorflow.python.keras.engine import training_arrays_v1
# from tensorflow.python.keras.layers import kernelized
from tensorflow.python.ops.linalg import matmul_registrations
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.compiler.xla import xla
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.data.util import nest
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.engine import input_spec
# from tensorflow.python.keras.feature_column import base_feature_layer
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.core import unsupported_features_checker
# from tensorflow.python.keras.premade import linear
# from tensorflow.python.keras.layers import local
from tensorflow.python.data.experimental.ops import map_defun
from tensorflow.python.framework import memory_checker
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.summary.writer import event_file_writer_v2
# from tensorflow.python.keras.preprocessing import text
from tensorflow.python.debug.cli import cli_shared
# from tensorflow.python.data.experimental.ops import threadpool
from tensorflow.python.autograph.converters import list_comprehensions
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.eager.memory_tests import memory_test_util
# from tensorflow.python.keras.layers import normalization
from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.training import training_util
from tensorflow.python.autograph.utils import misc
from tensorflow.python.ops.ragged import ragged_dispatch
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import control_flow_grad
# from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.client import session
from tensorflow.python.ops.numpy_ops import np_export
from tensorflow.python.framework import type_spec
# from tensorflow.python.data.experimental.ops import stats_ops
# from tensorflow.python.keras.distribute import optimizer_combinations
# from tensorflow.python.keras.layers.preprocessing import integer_lookup
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.autograph.pyct.testing import basic_definitions
# from tensorflow.python.ops import gen_tensor_forest_ops
from tensorflow.python.util import keras_deps
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.ops import gen_state_ops
# from tensorflow.python.keras.layers import multi_head_attention
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.keras.utils import io_utils
from tensorflow.python.keras.layers import pooling
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.training import adagrad
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import map_fn
from tensorflow.python.eager import test
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops.distributions import gamma
from tensorflow.python.autograph.pyct.testing import decorators
from tensorflow.python.framework import device
from tensorflow.python.framework import test_util
from tensorflow.python.ops import image_grad
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops.distributions import identity_bijector
from tensorflow.python.ops.linalg import linear_operator_zeros
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops.ragged import ragged_where_op
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.eager import backprop
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.training import saver
from tensorflow.python.saved_model import load
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import template
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.saving import saved_model_experimental
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.ops import summary_op_util
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import module_wrapper
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.types import core
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.keras import losses
from tensorflow.python.ops import variables
from tensorflow.python.saved_model.model_utils import export_output
# from tensorflow.python.keras.layers.preprocessing import index_lookup_v1
# from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.data.experimental.ops import enumerate_ops
# from tensorflow.python.debug.examples import debug_mnist
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.framework import op_def_library
from tensorflow.python.ops import cudnn_rnn_grad
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.ops.distributions import beta
from tensorflow.python.autograph.converters import lists
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.data.experimental.ops import iterator_ops
from tensorflow.python.keras.layers import merge
from tensorflow.python.summary.writer import writer
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.framework import registry
from tensorflow.python.debug.wrappers import local_cli_wrapper
# from tensorflow.python.keras.feature_column import dense_features
from tensorflow.python.autograph.core import converter
# from tensorflow.python.debug.examples.v1 import debug_errors
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import function
from tensorflow.python.framework import python_memory_checker
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.ops import linalg_ops
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.saved_model import save_options
from tensorflow.python.eager import remote
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.ops.distributions import bijector_impl
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import gen_string_ops
# from tensorflow.python.debug.examples.v2 import debug_mnist_v2
from tensorflow.python.ops import gen_functional_ops
# from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.framework import graph_io
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops import parsing_ops
# from tensorflow.python.keras.applications import densenet
from tensorflow.python.ops import map_ops
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.framework import op_def_registry
# from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.tools.api.generator import create_python_api
from tensorflow.python.ops import image_grad_test_base
from tensorflow.python.feature_column import sequence_feature_column
from tensorflow.python.ops import gen_candidate_sampling_ops
from tensorflow.python.training import proximal_gradient_descent
from tensorflow.python.eager import monitoring
# from tensorflow.python.keras.feature_column import sequence_feature_column
from tensorflow.python.training.tracking import util
from tensorflow.python.framework import importer
# from tensorflow.python.keras.datasets import reuters
# from tensorflow.python.distribute.parallel_device import saving
from tensorflow.python.ops.distributions import bijector_test_util
from tensorflow.python.ops import tensor_array_ops
# from tensorflow.python.keras.layers.preprocessing import table_utils
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.distributions import student_t
# from tensorflow.python.keras.layers.preprocessing import text_vectorization_v1
from tensorflow.python.debug.wrappers import dumping_wrapper
# from tensorflow.python.keras.distribute import keras_correctness_test_base
# from tensorflow.python.keras.applications import resnet
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.saved_model import save
from tensorflow.python.platform import app
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.ops import list_ops
# from tensorflow.python.keras.layers.preprocessing import discretization
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.ops import custom_gradient
from tensorflow.python.eager import wrap_function
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.ragged import ragged_functional_ops
# from tensorflow.python.keras.distribute import saved_model_test_base
from tensorflow.python.framework import op_callbacks
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.ops import default_gradient
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.summary.writer import event_file_writer
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.platform import device_context
# from tensorflow.python.util import memory
from tensorflow.python.ops import logging_ops
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.distribute import input_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.ops import ctc_ops
from tensorflow.python.training import adagrad_da

class TFArgument(Argument):
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TF_TENSOR,
                          ArgType.KERAS_TENSOR, ArgType.TF_VARIABLE]
    _dtypes = [
        tf.bfloat16,
        tf.bool,
        tf.complex128,
        tf.complex64,
        tf.double,
        tf.float16,
        tf.float32,
        tf.float64,
        tf.half,
        tf.int16,
        tf.int32,
        tf.int64,
        tf.int8,
        tf.uint8,
        tf.uint16,
        tf.uint32,
        tf.uint64,
    ]
    _support_types = [
        ArgType.TF_TENSOR,
        ArgType.TF_VARIABLE,
        ArgType.KERAS_TENSOR,
        ArgType.TF_DTYPE,
        ArgType.TF_OBJECT,
    ]

    def __init__(
        self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None
    ) -> None:
        if isinstance(dtype, str):
            dtype = self.str_to_dtype(dtype)
        shape = self.shape_to_list(shape)

        super().__init__(value, type)
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype

        self.tensor_zero_flag_type1 = False
        self.tensor_zero_flag_type2 = False
        """
        ADDED by NIMA
        """

        self.tensor_empty_flag_type1 = False 
        self.tensor_empty_flag_type2 = False 
        self.tensor_empty_flag_type3 = False 
        self.tensor_empty_flag_type4 = False 
        self.tensor_empty_flag_type5 = False 
        self.tensor_empty_flag_type6 = False
        self.tensor_empty_flag_type7 = False 
        
        """NAN tensor
        """
        self.nan_input_tensor = False
        self.nan_input_tensor_whole = False
        
        """ Scalar
        """
        self.scalar_input_flag = False

        """ Negative
        """
        
        """Large tensor
 
        """
        self.large_tensor_flag_type1 = False
        self.large_tensor_flag_type2 = False
        self.large_tensor_flag_type3 = False
        self.large_tensor_flag_type4 = False
        self.large_tensor_flag_type5 = False
        self.large_tensor_flag_type6 = False
        self.large_tensor_flag_type7 = False
        self.large_tensor_flag_type8 = False
        
        """Large tensor
 
        """
        self.make_tensor_neg1 = False
        self.make_tensor_neg2 = False
        self.make_tensor_neg3 = False
        self.make_tensor_neg4 = False
        self.make_tensor_neg5 = False
        self.make_tensor_neg6 = False
        self.make_tensor_neg7 = False
        self.make_tensor_neg8 = False
        self.make_tensor_neg9 = False
        self.make_tensor_neg10 = False
        self.make_tensor_neg11 = False
        self.make_tensor_neg12 = False
        self.make_tensor_neg13 = False
        self.make_tensor_neg14 = False
        self.make_tensor_neg15 = False
        
        """Non scalar flags
        """
        self.non_scalar_input_flag1 = False
        self.non_scalar_input_flag2 = False
        self.non_scalar_input_flag3 = False
        self.non_scalar_input_flag4 = False
        self.non_scalar_input_flag5 = False

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    @staticmethod
    def shape_to_list(shape):
        if shape is None:
            return None
        if not isinstance(shape, list):
            try:
                shape = shape.as_list()
            except:
                shape = list(shape)
            else:
                shape = list(shape)
        shape = [1 if x is None else x for x in shape]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if tf.is_tensor(x):
            if tf.keras.backend.is_keras_tensor(x):
                return ArgType.KERAS_TENSOR
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE

    def mutate_value_random(self) -> None:
        """Apply random value mutation."""
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            self.minv, self.maxv = self.random_tensor_value_range(self.dtype)
        elif self.type == ArgType.TF_DTYPE:
            self.value = TFArgument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)
            assert 0

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = self.mutate_int_value(new_shape[i], minv=0)

        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(0.0)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [TFArgument(1, ArgType.INT),
                          TFArgument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1, 3), randint(1, 3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert 0

    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.1

        def if_mutate_null():
            return random.random() < 0.1

        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive():
                return False
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            if random.random() < 0.01:
                self.value = []  # with a probability return an empty list
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TF_TENSOR:
            dtype = choice(self._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            if not if_mutate_null():
                return False
            new_type = choice(self._support_types + super()._support_types)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [TFArgument(
                    2, ArgType.INT), TFArgument(3, ArgType.INT)]
            elif new_type == ArgType.TF_TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(TFArgument._dtypes)
        return True

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.2

    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1

    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.2

    def mutate_bool_value(self) -> bool:
        return choice([True, False])

    def mutate_int_value(self, value, minv=None, maxv=None) -> int:
        if TFArgument.if_mutate_int_random():
            value = choice(self._int_values)
        else:
            value += randint(-2, 2)
        if minv is not None:
            value = max(minv, value)
        if maxv is not None:
            value = min(maxv, value)
        return value

    def mutate_str_value(self, value) -> str:
        if TFArgument.if_mutate_str_random():
            return choice(self._str_values)
        return value

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(TFArgument._dtypes)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [tf.int16, tf.int32, tf.int64]:
            return tf.int8
        elif dtype in [tf.float32, tf.float64]:
            return tf.float16
        elif dtype in [tf.complex128]:
            return tf.complex64
        return dtype

    @staticmethod
    def random_tensor_value_range(dtype):
        assert isinstance(dtype, tf.dtypes.DType)
        minv = 0
        maxv = 1
        if (
            dtype.is_floating
            or dtype.is_complex
            or dtype == tf.string
            or dtype == tf.bool
        ):
            pass
        elif "int64" in dtype.name or "int32" in dtype.name or "int16" in dtype.name:
            minv = 0 if "uint" in dtype.name else -(1 << 8)
            maxv = 1 << 8
        else:
            try:
                minv = dtype.min
                maxv = dtype.max
            except Exception as e:
                minv, maxv = 0, 1
        return minv, maxv

    def to_code_tensor(self, var_name, low_precision=False):
        dtype = self.dtype
        if low_precision:
            dtype = self.low_precision_dtype(dtype)
        shape = self.shape
        if dtype is None:
            assert 0
        code = ""
        var_tensor_name = f"{var_name}_tensor"
        values = [2**8, 
                        2**22, 
                        2**24, 
                        1012756988, 
                        498444555, 
                        545646544, 
                        536870912, 
                        36028797018963968, 
                        1250999896764, 
                        10 ** 6,
                        2**60-1,
                        1676240524292489355,
                        100000000,
                        1610637938,
                        1e38,
                        1e20,
                        65534,
                        8968073515812833920,
                        2 ** 31,
                        92233720368,
                        1610612736,
                        3046875451,
                        1048576,
                        2147483647]
        big_number = random.choice(values)

        if dtype.is_floating:
            ### neg
            if self.make_tensor_neg1:
                big_number = random.choice(values)
                value = -big_number
                code += f"{var_tensor_name} = tf.constant({value}, shape={shape}, dtype=tf.{dtype.name},)\n"

            elif self.make_tensor_neg2:
                big_number = random.choice(values)
                value = -big_number
                code += f"{var_tensor_name} = tf.constant({value}, shape={shape}, dtype=tf.{dtype.name},)\n"
                
            ### large
            elif self.large_tensor_flag_type1:
                value = big_number
                code += f"{var_tensor_name} = tf.random.uniform({shape}, dtype=tf.{dtype.name}, maxval={value})\n"
            elif self.large_tensor_flag_type2:
                value = big_number
                code += f"{var_tensor_name} = tf.constant({value}, shape={shape}, dtype=tf.{dtype.name},)\n"
            elif self.large_tensor_flag_type3:
                value = big_number
                code += f"{var_tensor_name} = tf.ones([9], dtype=tf.float32) * 999999\n"
            elif self.large_tensor_flag_type4:
                value = big_number
                code += f"{var_tensor_name} = tf.clip_by_value(tf.random.uniform([9], dtype=tf.float32), 1000, 9999)\n"
            elif self.large_tensor_flag_type5:
                value = big_number
                code += f"{var_tensor_name} = tf.random.normal([9], mean=10000, stddev=100)\n"
            elif self.large_tensor_flag_type6:
                value = big_number
                code += f"{var_tensor_name} = tf.random.normal([9], mean=1000, stddev=100)\n"
            elif self.large_tensor_flag_type7:
                value = big_number
                code += f"{var_tensor_name} =tf.random.uniform([9], minval=1000, maxval=10000, dtype=tf.float32)\n"
            elif self.large_tensor_flag_type8:
                value = big_number
                code += f"{var_tensor_name} = tf.constant({value}, shape={shape}, dtype=tf.{dtype.name},)\n"
        
            ### non scalar
            elif self.non_scalar_input_flag1:
                code += f"{var_tensor_name} = tf.constant([1], shape=[1, 1], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag2:
                code += f"{var_tensor_name} = tf.constant([1], shape=[1,1], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag3:
                code += f"{var_tensor_name} = tf.constant([[1, 2], [3, 4]], shape=[2, 2], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag4:
                code += f"{var_tensor_name} = tf.constant([1.5, 2.5], shape=[2, 1], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag5:
                code += f"{var_tensor_name} = tf.constant([-1.5, -2.5], shape=[1, 2], dtype=tf.{dtype.name},)\n"
            
            # SCALAR
            elif self.scalar_input_flag:
                big_number = random.choice(values)
                code += f"{var_tensor_name} = {big_number} \n"
                
            # NAN
            elif self.nan_input_tensor:
                code += f"{var_tensor_name} = tf.constant(float('nan'), shape={shape}, dtype=tf.{dtype.name})\n"
            elif self.nan_input_tensor_whole:
                code += f"{var_tensor_name} = float('nan')\n"

            # Empty
            elif self.tensor_empty_flag_type1:
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type2:
                value = 0
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type3:
                value = 0
                code += f"{var_tensor_name} = tf.constant(0, shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type4:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.tensor_empty_flag_type5:
                code += f"{var_tensor_name} = tf.constant(0, shape=[], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type6:
                code += f"{var_tensor_name} = tf.constant(0, shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type7:
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 0, 0, 0], dtype=tf.{dtype.name},)\n"
            else:
                code += f"{var_tensor_name} = tf.random.uniform({shape}, dtype=tf.{dtype.name})\n"
                
        elif dtype.is_complex:
            if self.make_tensor_neg1:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value1 = random.choice(values)
                value2 = random.choice(values)
                code += f"{var_tensor_name} = tf.complex(tf.constant({-value1}, shape={shape}, dtype=tf.{ftype}), tf.constant({-value2}, shape={shape}, dtype=tf.{ftype}))\n"
                
            elif self.make_tensor_neg2:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value1 = random.randint(18, 161)
                value2 = random.randint(1, 161)
                big_number = random.randint(18, 161)
                value = -big_number
                code += f"{var_tensor_name} = tf.complex(tf.constant({-value1}, shape={shape}, dtype=tf.{ftype}), tf.constant({-value2}, shape={shape}, dtype=tf.{ftype}))\n"
                
            
            elif self.scalar_input_flag:
                big_number = random.choice(values)
                code += f"{var_tensor_name} = {big_number} \n"
                
            # Empty
            elif self.tensor_empty_flag_type1:
                code += f"{var_tensor_name} = tf.complex(tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},))\n"
            elif self.tensor_empty_flag_type2:
                value = 0
                code += f"{var_tensor_name} = tf.complex(tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},))\n"
            elif self.tensor_empty_flag_type3:
                value = 0
                code += f"{var_tensor_name} = tf.complex(tf.constant(0, shape=[0], dtype=tf.{dtype.name},))\n"
            elif self.tensor_empty_flag_type4:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.tensor_empty_flag_type5:
                code += f"{var_tensor_name} = tf.complex(tf.constant(0, shape=[], dtype=tf.{dtype.name},))\n"
            elif self.tensor_empty_flag_type6:
                code += f"{var_tensor_name} = tf.complex(tf.constant(0, shape=[0], dtype=tf.{dtype.name},))\n"
            elif self.tensor_empty_flag_type7:
                code += f"{var_tensor_name} = tf.complex(tf.constant([], shape=[0, 0, 0, 0], dtype=tf.{dtype.name},))\n"

            elif self.non_scalar_input_flag1:
                code += f"{var_tensor_name} = tf.complex(tf.constant([1], shape=[2, 1], dtype=tf.{dtype.name},))\n"
            elif self.non_scalar_input_flag2:
                code += f"{var_tensor_name} = tf.complex(tf.constant([1], shape=[1,1], dtype=tf.{dtype.name},))\n"
            elif self.non_scalar_input_flag3:
                code += f"{var_tensor_name} = tf.complex(tf.constant([[1, 2], [3, 4]], shape=[2, 2], dtype=tf.{dtype.name},))\n"
            elif self.non_scalar_input_flag4:
                code += f"{var_tensor_name} = tf.complex(tf.constant([1.5, 2.5], shape=[3, 2], dtype=tf.{dtype.name},))\n"
            elif self.non_scalar_input_flag5:
                code += f"{var_tensor_name} = tf.complex(tf.constant([-1.5, -2.5], shape=[2, 2], dtype=tf.{dtype.name},))\n"

            elif self.nan_input_tensor:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                code += (
                    "%s = tf.complex(tf.constant(float('nan'), shape=%s, dtype=tf.%s,),"
                    "tf.constant(float('nan'), shape=%s, dtype=tf.%s,))\n"
                    % (var_tensor_name, shape, ftype, shape, ftype)
                )
            elif self.nan_input_tensor_whole:
                code += "%s = float('nan')\n" % (var_tensor_name)
            elif self.tensor_zero_flag_type1:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value = [0]
                code += (
                    "%s = tf.complex(tf.constant(%s, shape=[0], dtype=tf.%s,),"
                    "tf.constant(%s, shape=[0], dtype=tf.%s,))\n"
                    % (var_tensor_name, value, ftype, value, ftype)
                )
            elif self.tensor_zero_flag_type2:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value = 0
                code += (
                    "%s = tf.complex(tf.constant(%s, shape=[0], dtype=tf.%s,),"
                    "tf.constant(%s, shape=[0], dtype=tf.%s,))\n"
                    % (var_tensor_name, value, ftype, value, ftype)
                )
            elif self.tensor_empty_flag_type1:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value = []
                code += (
                    "%s = tf.complex(tf.constant(%s, shape=[0], dtype=tf.%s,),"
                    "tf.constant(%s, shape=[0], dtype=tf.%s,))\n"
                    % (var_tensor_name, value, ftype, value, ftype)
                )
            elif self.tensor_empty_flag_type1:
                code += "%s = [] \n" % (var_tensor_name)
                
                
            elif self.large_tensor_flag_type1:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value1 = random.choice(values)
                value2 = random.choice(values)
                code += (
                    "%s =tf.complex(tf.random.uniform(%s, dtype=tf.%s, maxval=%s),"
                    "tf.random.uniform(%s, dtype=tf.%s, maxval=%s))\n"
                    % (
                        var_tensor_name,
                        shape,
                        ftype,
                        value1,
                        shape,
                        ftype,
                        value2,
                    )
                )
            elif self.large_tensor_flag_type2:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                value1 = random.choice(values)
                value2 = random.choice(values)
                code += (
                    "%s = tf.complex(tf.constant(%s, shape=%s, dtype=tf.%s,),"
                    "tf.constant(%s, shape=%s, dtype=tf.%s,))\n"
                    % (
                        var_tensor_name,
                        value1,
                        shape,
                        ftype,
                        value2,
                        shape,
                        ftype,
                    )
                )
            else:
                ftype = "float64" if dtype == tf.complex128 else "float32"
                code += (
                    "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s),"
                    "tf.random.uniform(%s, dtype=tf.%s))\n"
                    % (var_tensor_name, shape, ftype, shape, ftype)
                )
        elif dtype == tf.bool:
            
            if self.tensor_empty_flag_type1:
                value = []
                code += "%s = tf.constant(%s, shape=[0,0], dtype=tf.%s,)\n" % (
                    var_tensor_name,
                    value,
                    dtype.name,
                )

            ### large
            elif self.large_tensor_flag_type1:
                d1 = random.choice(values)
                d2 = random.choice(values)
                val = random.choice([True, False])
                code += f"{var_tensor_name} = tf.constant({val}, shape=[{d1},{d2}])\n"   
            
            ### empty
            elif self.tensor_empty_flag_type1:
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type2:
                value = 0
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 0, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type3:
                code += f"{var_tensor_name} = tf.constant([], shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type4:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.tensor_empty_flag_type5:
                code += f"{var_tensor_name} = tf.constant(0, shape=[], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type6:
                code += f"{var_tensor_name} = tf.constant(0, shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type7:
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 0, 0, 0], dtype=tf.{dtype.name},)\n"
            
            # Non scalar
            elif self.non_scalar_input_flag1:
                value = False
                code += f"{var_tensor_name} = tf.constant({value}, shape=[4, 1], dtype=tf.bool,)\n"
            elif self.non_scalar_input_flag2:
                value = False
                code += f"{var_tensor_name} = tf.constant({value}, shape=[1,1], dtype=tf.bool,)\n"
            elif self.non_scalar_input_flag3:
                value = False
                code += f"{var_tensor_name} = tf.constant([{value}, {value}], shape=[2, 2], dtype=tf.bool,)\n"
            elif self.non_scalar_input_flag4:
                value = False
                code += f"{var_tensor_name} = tf.constant([{value}, {value}], shape=[2, 6], dtype=tf.bool,)\n"
            elif self.non_scalar_input_flag5:
                value = False
                code += f"{var_tensor_name} = tf.constant([{value}, {value}], shape=[3, 8], dtype=tf.bool,)\n"
                
            elif self.tensor_empty_flag_type1:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.scalar_input_flag:
                value = random.randint(1, 1000)
                code += f"{var_tensor_name} = {value} \n"
            else:
                code += (
                    "%s = tf.cast(tf.random.uniform("
                    "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n"
                    % (var_tensor_name, shape)
                )
        elif dtype == tf.string:
            code += "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (
                var_tensor_name,
                shape,
            )
        elif dtype in [tf.int32, tf.int64]:
            
            if self.make_tensor_neg1:
                value = random.randint(1879048192, 161063793887434)
                code += "%s = tf.constant(%s, shape=%s, dtype=tf.%s,)\n" % (
                    var_tensor_name,
                    -value,
                    shape,
                    dtype.name,
                )

            elif self.make_tensor_neg2:
                value = random.randint(1, 161)
                code += "%s = tf.constant(%s, shape=%s, dtype=tf.%s,)\n" % (
                    var_tensor_name,
                    -value,
                    shape,
                    dtype.name,
                )
                
            elif self.large_tensor_flag_type1:
                value = random.randint(1879048192, 161063793887434)
                code += "%s = tf.random.uniform(%s, dtype=tf.%s, maxval=%s)\n" % (
                    var_tensor_name,
                    shape,
                    dtype.name,
                    abs(value),
                )
            elif self.large_tensor_flag_type2:
                value = random.randint(1879048192, 161063793887434)
                code += "%s = tf.constant(%s, shape=%s, dtype=tf.%s,)\n" % (
                    var_tensor_name,
                    abs(value),
                    shape,
                    dtype.name,
                )

            ### empty
            elif self.tensor_empty_flag_type1:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type2:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant(0, shape=[0, 0, 1], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type3:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant([], shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type4:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.tensor_empty_flag_type5:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant(0, shape=[], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type6:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant(0, shape=[0], dtype=tf.{dtype.name},)\n"
            elif self.tensor_empty_flag_type7:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.constant([], shape=[0, 0, 0, 0], dtype=tf.{dtype.name},)\n"

            # large tensor
            elif self.large_tensor_flag_type1:
                value = big_number
                code += f"{var_tensor_name} = tf.constant(9999)\n"
            elif self.large_tensor_flag_type2:
                value = big_number
                code += f"{var_tensor_name} = tf.fill([9], 100)\n"
            elif self.large_tensor_flag_type3:
                value = big_number
                code += f"{var_tensor_name} = tf.ones([9], dtype=tf.{dtype.name}) * 999999\n"
            elif self.large_tensor_flag_type4:
                value = big_number
                code += f"{var_tensor_name} = tf.clip_by_value(tf.random.uniform([9], dtype=tf.{dtype.name}), 1000, 9999)\n"
            elif self.large_tensor_flag_type5:
                value = big_number
                code += f"{var_tensor_name} = tf.fill([9], {big_number})\n"
            elif self.large_tensor_flag_type6:
                value = big_number
                code += f"{var_tensor_name} = tf.fill([5], {big_number})\n"
            elif self.large_tensor_flag_type7:
                value = big_number
                code += f"{var_tensor_name} =tf.random.uniform([9], minval=1000, maxval=10000, dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type8:
                value = big_number
                code += f"{var_tensor_name} = tf.constant({value}, shape={shape}, dtype=tf.{dtype.name},)\n"
            
            # non scalar
            elif self.non_scalar_input_flag1:
                code += f"{var_tensor_name} = tf.constant([1], shape=[3, 1], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag2:
                code += f"{var_tensor_name} = tf.constant([1], shape=[1,1], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag3:
                code += f"{var_tensor_name} = tf.constant([[1, 2], [3, 4]], shape=[2, 2], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag4:
                code += f"{var_tensor_name} = tf.constant([1.5, 2.5], shape=[2, 2], dtype=tf.{dtype.name},)\n"
            elif self.non_scalar_input_flag5:
                code += f"{var_tensor_name} = tf.constant([-1.5, -2.5], shape=[2, 1], dtype=tf.{dtype.name},)\n"
                
            elif self.scalar_input_flag:
                value = random.choice(values)
                code += f"{var_tensor_name} = {value} \n"
            elif self.nan_input_tensor:
                code += "%s = tf.constant(float('nan'), shape=%s, dtype=tf.float64)\n" % (
                    var_tensor_name,
                    shape,
                )
            elif self.nan_input_tensor_whole:
                code += "%s = float('nan')\n" % (var_tensor_name)
            elif self.tensor_zero_flag_type1:
                value = [0.0000000000000]
                code += "%s = tf.constant(%s, shape=[0], dtype=tf.%s)\n" % (
                    var_tensor_name,
                    value,
                    dtype.name,
                )
            elif self.tensor_zero_flag_type2:
                value = 0.0000000000000
                code += "%s = tf.constant(%s, shape=[0], dtype=tf.%s)\n" % (
                    var_tensor_name,
                    value,
                    dtype.name,
                )
            elif self.tensor_empty_flag_type1:
                value = []
                code += "%s = tf.constant(%s, shape=[0], dtype=tf.%s)\n" % (
                    var_tensor_name,
                    value,
                    dtype.name,
                )
            elif self.tensor_empty_flag_type1:
                code += "%s = [] \n" % (var_tensor_name)

            else:
                code += (
                    "%s = tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.%s)\n"
                    % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
                )
        else:
            if self.make_tensor_neg1:
                value = random.choice(values)
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(%s, shape=%s, dtype=tf.int64,),"
                    "dtype=tf.%s)\n" % (
                        var_tensor_name, -value, shape, dtype.name)
                )
    
            elif self.make_tensor_neg2:
                value = random.randint(18, 161)
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(%s, shape=%s, dtype=tf.int64,),"
                    "dtype=tf.%s)\n" % (
                        var_tensor_name, -value, shape, dtype.name)
                )
            
            
            ### empty
            elif self.tensor_empty_flag_type1:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant([], shape=[0, 1, 1], dtype=tf.{dtype.name}), dtype=tf.{dtype.name})\n"
            elif self.tensor_empty_flag_type2:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant(0, shape=[0, 0, 1], dtype=tf.{dtype.name}), dtype=tf.{dtype.name}) \n"
            elif self.tensor_empty_flag_type3:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant([], shape=[0], dtype=tf.{dtype.name}), dtype=tf.{dtype.name}))\n"
            elif self.tensor_empty_flag_type4:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.tensor_empty_flag_type5:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant(0, shape=[], dtype=tf.{dtype.name}), dtype=tf.{dtype.name})\n"
            elif self.tensor_empty_flag_type6:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant(0, shape=[0], dtype=tf.{dtype.name}), dtype=tf.{dtype.name})\n"
            elif self.tensor_empty_flag_type7:
                value = random.randint(1, 10)
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant([], shape=[0, 0, 0, 0], dtype=tf.{dtype.name}),dtype=tf.{dtype.name}))\n"
            
            # large
        
            elif self.large_tensor_flag_type1:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant(9999), dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type2:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.fill([9], 100), dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type3:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.ones([9]) * 999999, dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type4:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.clip_by_value(tf.random.uniform([9]), 1000, 9999), dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type5:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.fill([9], {big_number}), dtype={dtype.name})\n"
            elif self.large_tensor_flag_type6:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.fill([5], {big_number}), dtype={dtype.name})\n"
            elif self.large_tensor_flag_type7:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.random.uniform([9], minval=1000, maxval=10000, dtype=tf.{dtype.name}), dtype=tf.{dtype.name})\n"
            elif self.large_tensor_flag_type8:
                value = big_number
                code += f"{var_tensor_name} = tf.saturate_cast(tf.constant({value}, shape={shape}, dtype=tf.{dtype.name}), dtype=tf.{dtype.name})\n"
            
            
            # non scalar
            elif self.non_scalar_input_flag1:
                code += (f"{var_tensor_name} = tf.saturate_cast(tf.constant([1], shape=[2, 1], dtype=tf.{dtype.name},))\n")
            elif self.non_scalar_input_flag2:
                code += (f"{var_tensor_name} = tf.saturate_cast(tf.constant([1], shape=[1,1], dtype=tf.{dtype.name},))\n")
            elif self.non_scalar_input_flag3:
                code += (f"{var_tensor_name}  = tf.saturate_cast(tf.constant([[1, 2], [3, 4]], shape=[2, 2], dtype=tf.{dtype.name},))\n")
            elif self.non_scalar_input_flag4:
                code += (f"{var_tensor_name}  = tf.saturate_cast(tf.constant([1.5, 2.5], shape=[2, 2], dtype=tf.{dtype.name},))\n")
            elif self.non_scalar_input_flag5:
                code += (f"{var_tensor_name}  = tf.saturate_cast(tf.constant([-1.5, -2.5], shape=[2, 1], dtype=tf.{dtype.name},))\n")
                
            elif self.scalar_input_flag:
                value = random.randint(1879048192, 161063793887434)
                code += f"{var_tensor_name} = {value} \n"
            elif self.nan_input_tensor:
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(float('nan'), shape=%s, dtype=tf.float64,),"
                    "dtype=tf.%s)\n" % (var_tensor_name, shape, dtype.name)
                )
            elif self.nan_input_tensor_whole:
                code += "%s = float('nan')\n" % (var_tensor_name)
            elif self.tensor_zero_flag_type1:
                value = [0]
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(%s, shape=%s, dtype=tf.int64,),"
                    "dtype=tf.%s)\n" % (
                        var_tensor_name, value, shape, dtype.name)
                )

            elif self.tensor_empty_flag_type1:
                value = []
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(%s, shape=[0], dtype=tf.int64,),"
                    "dtype=tf.%s)\n" % (var_tensor_name, value, dtype.name)
                )
            elif self.tensor_empty_flag_type2:
                code += "%s = [] \n" % (var_tensor_name)
            elif self.large_tensor_flag_type1:
                value = random.choice(values)
                code += (
                    "%s = tf.saturate_cast("
                    "tf.random.uniform(%s, dtype=tf.int64, maxval=%s),"
                    "dtype=tf.%s)\n" % (
                        var_tensor_name, shape, abs(value), dtype.name)
                )
            elif self.large_tensor_flag_type2:
                value = random.choice(values)
                code += (
                    "%s = tf.saturate_cast("
                    "tf.constant(%s, shape=%s, dtype=tf.int64,),"
                    "dtype=tf.%s)\n" % (
                        var_tensor_name, abs(value), shape, dtype.name)
                )
            else:
                code += (
                    "%s = tf.saturate_cast("
                    "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), "
                    "dtype=tf.%s)\n"
                    % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
                )
        code += f"{var_name} = tf.identity({var_tensor_name})\n"
        return code

    def to_code_keras_tensor(self, var_name, low_precision=False):
        return self.to_code_tensor(var_name, low_precision=low_precision)

    def to_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            # Did not consider cloning for in-place operation here.
            code = ""
            if self.type == ArgType.TF_TENSOR:
                code = self.to_code_tensor(
                    var_name, low_precision=low_precision)
            elif self.type == ArgType.TF_VARIABLE:
                code = self.to_code_tensor(
                    var_name, low_precision=low_precision)
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            elif self.type == ArgType.KERAS_TENSOR:
                code = self.to_code_keras_tensor(
                    var_name, low_precision=low_precision)
            return code
        return super().to_code(var_name)

    def to_diff_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(
                    f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            code = f"{var_name} = tf.identity({var_name}_tensor)\n"
            if not low_precision:
                code += f"{var_name} = tf.cast({var_name}, tf.{self.dtype.name})\n"
            if self.type == ArgType.TF_VARIABLE:
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            return code
        return ""

    def new_mutation_multiple(self, RULE=None):
        if RULE == "LARGE_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(large=True)
        elif RULE == "NEGATIVE_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(neg=True)
        elif RULE == "NEGATIVE_LARGE_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(neg_large=True)
        elif RULE == "ZERO_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(zero=True)
        elif RULE == "EMPTY_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(empty=True)
        elif RULE == "NAN_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(nan=True)
        elif RULE == "NONE_INTEGER":
            if self.type == ArgType.INT: 
                self.mutate_integer(none=True)
        elif RULE == "LARGE_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(large=True)
        elif RULE == "NEGATIVE_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(neg=True)     
        elif RULE == "NEGATIVE_LARGE_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(neg_large=True)
        elif RULE == "ZERO_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(zero=True)
        elif RULE == "EMPTY_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(empty=True)
        elif RULE == "NAN_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_float(nan=True)
        elif RULE == "NONE_FLOAT":
            if self.type == ArgType.FLOAT: 
                self.mutate_integer(none=True)
        elif RULE == "INVALID_STRING":
            if self.type == ArgType.STR: 
                self.mutate_str(invalid=True)
        elif RULE == "EMPTY_STRING1":
            if self.type == ArgType.STR:
                self.mutate_str(empty1=True)
        elif RULE == "EMPTY_STRING2":
            if self.type == ArgType.STR:
                self.mutate_str(empty2=True)
        elif RULE == "NAN_STRING":
            if self.type == ArgType.STR:
                self.mutate_str(nan=True)
        elif RULE == "NONE_STRING":
            if self.type == ArgType.STR:
                self.mutate_str(none=True)
        elif RULE == "RANK_REDUCTION_EXPANSION":
            self.modify_rank()
        elif RULE == "EMPTY_TENSOR_TYPE1":
            self.make_tensor_empty_type1()
        elif RULE == "EMPTY_TENSOR_TYPE2":
            self.make_tensor_empty_type2()
        elif RULE == "EMPTY_LIST":
            self.make_list_tuple_empty()
        elif RULE == "LARGE_TENSOR_TYPE1":
            self.make_tensor_large_type1()
        elif RULE == "LARGE_TENSOR_TYPE2":
            self.make_tensor_large_type2()
        elif RULE == "LARGE_LIST_ELEMENT":
            self.make_list_element_large()
        elif RULE == "ZERO_TENSOR_TYPE1":
            self.make_tensor_zero_type1()
        elif RULE == "ZERO_TENSOR_TYPE2":
            self.make_tensor_zero_type2()
        elif RULE == "NAN_TENSOR":
            self.make_tensor_nan()
        elif RULE == "NAN_TENSOR_WHOLE":
            self.make_tensor_nan_whole()
        elif RULE == "NON_SCALAR_INPUT":
            self.make_input_non_scalar()
        elif RULE == "SCALAR_INPUT":
            self.make_input_scalar()
        else:
            return

    """
    ######################### Delete list element ##############
    #################################################################
    """

    def increase_integer(self, value) -> int:
        new_value = random.randint(100, 1000)
        val = -new_value
        return val

    def new_mutation(self, param_space, param_index, param_name):
        if self.type == ArgType.INT:
            self.value = self.increase_integer(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_negative(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.make_bool_inverse(self.value)
        elif self.type == ArgType.STR:
            self.value = float('nan')
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.new_mutation(param_space, param_index, param_name)
        elif self.type in self._tensor_arg_dtypes:
            self.modify_rank(param_space, param_index, param_name)
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return

    def mutate_value(self):
        self.mutate_value_random()

    """
    ######################### NON SCALAR INPUTS ##############
    #################################################################
    """

    def make_input_scalar(self):
        if self.type in self._tensor_arg_dtypes:
            self.scalar_input_flag = True

    """
    ######################### NON SCALAR INPUTS ##############
    #################################################################
    """

    def make_input_non_scalar(self):
        if self.type == ArgType.TF_TENSOR:
            super().activate_non_scalar_input_flag()
        else:
            return

    """
    ######################### NAN INPUT TO TENSORS ##############
    #################################################################
    """

    def make_tensor_nan(self):
        if self.type in self._tensor_arg_dtypes:
            self.nan_input_tensor = True

    def make_tensor_nan_whole(self):
        if self.type in self._tensor_arg_dtypes:
            self.nan_input_tensor_whole = True

    """
    ######################### ZERO ELEMENTS OR TENSORS ##############
    #################################################################
    """

    def make_tensor_zero_type1(self):
        if self.type in self._tensor_arg_dtypes:
            self.tensor_zero_flag_type1 = True

    def make_tensor_zero_type2(self):
        if self.type in self._tensor_arg_dtypes:
            self.tensor_zero_flag_type2 = True

    """
    ######################### VERY LARGE INPUTS #####################
    #################################################################
    """

    def make_list_element_large(self):
        if self.type == ArgType.INT:
            self.value = 125091515651
        elif self.type == ArgType.FLOAT:
            self.value = 2251.000000
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.make_list_element_large()
        else:
            return

    def make_tensor_large_type1(self):
        if self.type in self._tensor_arg_dtypes:
            self.large_tensor_flag_type1 = True

    def make_tensor_large_type2(self):
        if self.type in self._tensor_arg_dtypes:
            self.large_tensor_flag_type2 = True

    """
    ######################## EMPTY INPUTS ###########################
    ################################################################# 
    """

    def make_int_zero(self, value) -> int:
        new_value = 0
        return new_value

    def make_float_zero(self, value) -> float:
        new_value = 0
        return new_value

    def make_tensor_empty_type1(self):
        if self.type in self._tensor_arg_dtypes:
            self.tensor_empty_flag_type1 = True

    def make_tensor_empty_type2(self):
        if self.type in self._tensor_arg_dtypes:
            self.tensor_empty_flag_type2 = True

    def make_list_tuple_empty(self):
        if self.type == ArgType.INT:
            self.value = self.make_int_zero(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_zero(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.make_list_tuple_empty()
        else:
            return

    """
    ######################## TENSOR RANK REDUCTION EXPANSION#########
    #################################################################
    """

    def alter_tensor_shape(self, old_shape, op):
        new_shape = old_shape
        # Change rank
        max_n = random.randint(1, len(self.shape)+1)
        if op:
            new_shape.pop()
        else:
            for i in range(max_n):
                new_shape.append(max_n)

        # if change_dim():
        #     RandomIndex = randint(0, len(new_shape))
        #     new_shape[RandomIndex] = choice(rand_dims)
        return new_shape

    def random_rank(self) -> None:
        random_rank = []
        rank_len = random.randint(1, len(self.shape)+1)
        for i in range(0, rank_len):
            n = random.randint(1, 10)
            random_rank.append(n)
        return random_rank

    def modify_rank(self, param_space, param_index, param_name) -> None:
        if self.type in self._tensor_arg_dtypes:
            if param_index == 1:
                param_name = f"parameter:{param_index-1}"
            if param_space[param_name].MISMATCH_OP == 'EXPANSION':
                op = True
            elif param_space[param_name].MISMATCH_OP == 'REDUCTION':
                op = False
            else:
                op = random.choice([True, False])
            self.shape = self.alter_tensor_shape(self.shape, op)
            if op:
                self.MISMATCH_OP = "REDUCTION"
            else:
                self.MISMATCH_OP = "EXPANSION"
        else:
            return

    """
    ######################## NEGATIVE MUTATION ########################
    #################################################################
    """

    def make_int_negative(self, value) -> int:
        value = random.randint(1, 100)
        new_value = -value
        return new_value

    def make_float_negative(self, value) -> float:
        value = random.uniform(0, 1)
        new_value = -value
        return new_value

    def make_tensor_negative(self) -> None:
        self.make_tensor_neg = True

    def make_bool_inverse(self, value) -> bool:
        return not value

    def mutate_preemptives(self) -> None:
        if self.type == ArgType.INT:
            self.value = float('nan')
        elif self.type == ArgType.FLOAT:
            self.value = float('nan')
        elif self.type == ArgType.STR:
            self.value = float('nan')
        elif self.type == ArgType.BOOL:
            self.value = float('nan')
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.mutate_preemptives()
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return

    def mutate_negative(self) -> None:
        if self.type == ArgType.INT:
            self.value = self.make_int_negative(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_negative(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.make_bool_inverse(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.mutate_negative()
        elif self.type in self._tensor_arg_dtypes:
            self.make_tensor_negative()
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return

    """
    #################################################################
    #################################################################
    """
################################################ NEW implementation ########################################

    def mutate_bool(self,random_=False, none=False, nan=False, empty=False, zero=False):
        if nan:
            self.value = float('nan')
        elif empty:
            self.value = [()]
        elif zero:
            self.value = 0.0
        elif none:
            self.value = None
        elif random_:
            b_ = random.choice([True, False])
            self.value = b_
        else:
            return
        
    def mutate_integer(self, zero=False, large=False, neg=False, neg_large=False, nan=False, none=False, empty=False) -> int:
            if zero:
                self.value = 0.0
            elif large:
                values = [2**8, 
                        2**22, 
                        2**24, 
                        1012756988, 
                        498444555, 
                        545646544, 
                        536870912, 
                        36028797018963968, 
                        1250999896764, 
                        10 ** 6,
                        2**60-1,
                        1676240524292489355,
                        100000000,
                        1610637938,
                        1e38,
                        1e20,
                        65534,
                        8968073515812833920,
                        2 ** 31,
                        92233720368,
                        1610612736,
                        3046875451,
                        1048576,
                        2147483647]
                new_value = random.choice(values)
                self.value = new_value
            elif neg:
                new_value = random.randint(1e3, 1e5)
                self.value = -new_value
            elif neg_large:
                values = [2**8, 
                        2**22, 
                        2**24, 
                        1012756988, 
                        498444555, 
                        545646544, 
                        536870912, 
                        36028797018963968, 
                        1250999896764, 
                        10 ** 6,
                        2**60-1,
                        1676240524292489355,
                        100000000,
                        1610637938,
                        1e38,
                        1e20,
                        65534,
                        8968073515812833920,
                        2 ** 31,
                        92233720368,
                        1610612736,
                        3046875451,
                        1048576,
                        2147483647]
                new_value = random.choice(values)
                self.value = -new_value
            elif nan:
                self.value = float('nan')
            elif none:
                self.value = None
            elif empty:
                self.value = [()]
            else:
                new_value = random.randint(1e3, 1e5)
                val_ = -new_value
                self.value = val_
   
    def mutate_float(self, zero=False, large=False, neg=False, neg_large=False, nan=False, none=False, empty=False) -> float:
        if zero:
            self.value = 0.0
        elif large:
            new_value = random.choice([3.402823e+38, 1.986e+67])
            self.value = new_value
        elif neg_large:
            value = [1250999996764.1,
                    10000000000000.0,
                    1.7976931348623157e+308,
                    9007199254740992.0,
                    0.0,
                    12345678901234.56,
                    1.4013e-45,
                    9.88131e-324,
                    1.17549435082e-38,
                    3.402823466e+38,
                    1.4013e-45,
                    1111111111111111.1,
                    2.2250738585072014e-308,
                    4444444444444.44,
                    123456789123.321,
                    3.141592653589793,
                    2.718281828459045,
                    9876543210.123456,
                    1010101010.10101,
                    1717171717.1717172,
                    2.7182818284590455,
                    3007199254740992.7,
                    41421356237309515,
                    7171717171717.717,
                    78964512365478965.22,
                    999999999999999999.2]
            new_value = random.choice(value)
            self.value = - new_value
        elif neg:
            values = [0.1,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                10.0,
                50.0,
                100.0,
                500.0,
                1000.0,
                1e05,
                0.0001,
                0.001,
                0.01,
                1e10,
                1e15,
                1e20,
                1e25,
                1e30,
                1e35,
                1e40,
                1.23456789,
                999999999.9999999,
                1.7976931348623157e+308,
                2.2250738585072014e308]
            new_value = random.choice(values)
            self.value = -new_value
        elif nan:
            self.value = float('nan')
        elif none:
            self.value = None
        elif empty:
            self.value = [[]]
        else:
            new_value = random.random()
            val_ = -new_value
            self.value = val_
    
    def mutate_str(self, invalid=False, empty1=False, nan=False, none=False, empty2=False) -> str:
        if invalid:
            non_ascii_list = [chr(i) for i in range(128, 256)]
            # def generate_random_word(length):
            #     return ''.join(random.choice(non_ascii_list) for _ in range(length))
            # random_words = [generate_random_word(random.randint(5, 10)) for _ in range(5)]
            # value = random.choice(random_words)
            self.value = "(0)"
        if nan:
            self.value = float('nan')
        elif none:
            self.value = None
        elif empty1:
            self.value = []
        elif empty2:
            self.value = "(0)"

    def modify_tensor_rank(self, large=False, neg=False, zero=False, empty=False, neg_large=False):
        if large:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    self.shape[i] = random.randint(20, 50)
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, large=True)
        elif neg:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    new_val = random.randint(1, 10)
                    self.shape[i] = -new_val
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, neg_large=True)
        elif zero:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    self.shape[i] = 0
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, zero=True)
        elif empty:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                self.shape = []
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, empty=True)
        else:
            return

    def each_arg_mutation(self, partition):
        # TENSOR
        if partition == 'NULL':
            return
        if partition == 'NULL_TF_OBJ':
            return
        if partition == 'NON_SCALAR_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'non_scalar_input_flag1', True),
                lambda: setattr(self, 'non_scalar_input_flag2', True),
                lambda: setattr(self, 'non_scalar_input_flag3', True),
                lambda: setattr(self, 'non_scalar_input_flag4', True),
                lambda: setattr(self, 'non_scalar_input_flag5', True)
            ]

            selected_op = random.choice(ops)
            selected_op()
        if partition == 'LARGE_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'large_tensor_flag_type1', True),
                lambda: setattr(self, 'large_tensor_flag_type2', True),
                lambda: setattr(self, 'large_tensor_flag_type3', True),
                lambda: setattr(self, 'large_tensor_flag_type4', True),
                lambda: setattr(self, 'large_tensor_flag_type5', True),
                lambda: setattr(self, 'large_tensor_flag_type6', True),
                lambda: setattr(self, 'large_tensor_flag_type7', True),
                lambda: setattr(self, 'large_tensor_flag_type8', True)
            ]
        
            selected_op = random.choice(ops)
            selected_op()
        if partition == 'NEGATIVE_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'make_tensor_neg1', True),
                lambda: setattr(self, 'make_tensor_neg2', True),
            ]
        
            selected_op = random.choice(ops)
            selected_op()
        if partition == 'SCALAR_INPUT_TENSOR':
            self.scalar_input_flag = True
        if partition == 'NAN_INPUT_TENSOR':
            self.nan_input_tensor = True
        if partition == 'NAN_INPUT_TENSOR_WHOLE':
            self.nan_input_tensor_whole = True
        if partition == 'TENSOR_EMPTY_FLAG':
            ops = [
                lambda: setattr(self, 'tensor_empty_flag_type1', True),
                lambda: setattr(self, 'tensor_empty_flag_type2', True),
                lambda: setattr(self, 'tensor_empty_flag_type3', True),
                lambda: setattr(self, 'tensor_empty_flag_type4', True),
                lambda: setattr(self, 'tensor_empty_flag_type5', True),
                lambda: setattr(self, 'tensor_empty_flag_type6', True),
                lambda: setattr(self, 'tensor_empty_flag_type7', True),
            ]
            
    
        
            selected_op = random.choice(ops)
            selected_op()

        # LIST
        if partition == 'LARGE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(large=True)
                else:
                    return
        if partition == 'NEGATIVE_LARGE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(neg_large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg_large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg_large=True)
                else:
                    return
        if partition == 'ZERO_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(zero=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(zero=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(zero=True)
                else:
                    return
        if partition == 'NEGATIVE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(neg=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg=True)
                else:
                    return
        if partition == 'EMPTY_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = ['']
                elif self.type == ArgType.INT:
                    self.value = ['']
                elif self.type == ArgType.FLOAT:
                    self.value = ['']
                elif self.type == ArgType.STR:
                    self.value = ['']
                else:
                    return
        if partition == 'INVALID_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.STR:
                    self.mutate_str(invalid=True)
        if partition == 'NONE_INPUT_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = None
                elif self.type == ArgType.INT:
                    self.value = None
                elif self.type == ArgType.FLOAT:
                    self.value = None
                elif self.type == ArgType.STR:
                    self.value = None
                else:
                    return
        if partition == 'NAN_INPUT_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = float('nan')
                elif self.type == ArgType.INT:
                    self.value = float('nan')
                elif self.type == ArgType.FLOAT:
                    self.value = float('nan')
                elif self.type == ArgType.STR:
                    self.value = float('nan')
                else:
                    return
        # TUPLE
        if partition == 'LARGE_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(large=True)
                else:
                    return
        if partition == 'ZERO_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(zero=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(zero=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(zero=True)
                else:
                    return
        if partition == 'NEGATIVE_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.modify_tensor_rank(neg=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg=True)
                else:
                    return
        if partition == 'EMPTY_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = ['']
                elif self.type == ArgType.INT:
                    self.value = ['']
                elif self.type == ArgType.FLOAT:
                    self.value = ['']
                elif self.type == ArgType.STR:
                    self.value = ['']
                else:
                    return
        if partition == 'INVALID_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.STR:
                    self.mutate_str(invalid=True)
        if partition == 'NONE_INPUT_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = None
                elif self.type == ArgType.INT:
                    self.value = None
                elif self.type == ArgType.FLOAT:
                    self.value = None
                elif self.type == ArgType.STR:
                    self.value = None
                else:
                    return
        if partition == 'NAN_INPUT_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TF_TENSOR:
                    self.value = float('nan')
                elif self.type == ArgType.INT:
                    self.value = float('nan')
                elif self.type == ArgType.FLOAT:
                    self.value = float('nan')
                elif self.type == ArgType.STR:
                    self.value = float('nan')
                else:
                    return

        # INT
        if partition == 'NEGATIVE_INTEGERS':
            self.mutate_integer(neg=True)
        if partition == 'ZERO_INTEGER':
            self.mutate_integer(zero=True)
        if partition == 'LARGE_INTEGER':
            self.mutate_integer(large=True)
        if partition == 'NEGATIVE_LARGE_INTEGER':
            self.mutate_integer(neg_large=True)
        if partition == 'EMPTY':
            self.mutate_integer(empty=True)
        if partition == 'NONE':
            self.mutate_integer(none=True)
        if partition == 'NAN':
            self.mutate_integer(nan=True)

        # FLOAT
        if partition == 'NEGATIVE_FLOAT':
            self.mutate_float(neg=True)
        if partition == 'ZERO_FLOAT':
            self.mutate_float(zero=True)
        if partition == 'LARGE_FLOAT':
            self.mutate_float(large=True)
        if partition == 'NEGATIVE_LARGE_FLOAT':
            self.mutate_float(neg_large=True)
        if partition == 'EMPTY':
            self.mutate_float(empty=True)
        if partition == 'NONE':
            self.mutate_float(none=True)
        if partition == 'NAN':
            self.mutate_float(nan=True)

        # STRING

        if partition == 'INVALID_STRING':
            self.mutate_str(invalid=True)
        if partition == 'EMPTY_STRING':
            self.mutate_str(empty1=True)
        if partition == 'EMPTY':
            self.mutate_str(empty2=True)
        if partition == 'NONE':
            self.mutate_str(none=True)
        if partition == 'NAN':
            self.mutate_str(nan=True)
            
        # BOOL

        if partition == 'RANDOM_BOOL':
            self.mutate_bool(random_=True)
        if partition == 'NONE_BOOL':
            self.mutate_bool(none=True)
        if partition == 'NAN_BOOL':
            self.mutate_bool(nan=True)
        if partition == 'EMPTY_BOOL':
            self.mutate_bool(empty=True)
        if partition == 'ZERO_BOOL':
            self.mutate_bool(zero=True)
        
    """
    ##################################################################
    ##################################################################
    """

    @staticmethod
    def generate_arg_from_signature(signature):
        if isinstance(signature, bool):
            return TFArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TFArgument(signature, ArgType.INT)
        if isinstance(signature, float):
            return TFArgument(signature, ArgType.FLOAT)
        if isinstance(signature, str):
            return TFArgument(signature, ArgType.STR)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.LIST)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.TUPLE)

        if not isinstance(signature, dict):
            return TFArgument(None, ArgType.NULL)

        if "type" not in signature and "Label" not in signature:
            return TFArgument(None, ArgType.NULL)

        label = signature["type"] if "type" in signature else signature["Label"]

        if label == "tf_object":
            if "class_name" not in signature:
                return TFArgument(None, ArgType.TF_OBJECT)

            if (
                signature["class_name"]
                == "tensorflow.python.keras.engine.keras_tensor.KerasTensor"
                or signature["class_name"]
                == "tensorflow.python.ops.variables.RefVariable"
            ):
                dtype = signature["dtype"]
                shape = signature["shape"]
                dtype = TFArgument.str_to_dtype(dtype)
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
                name = signature["to_str"].replace(
                    "<dtype: '", "").replace("'>", "")
                value = eval("tf." + name)
                return TFArgument(value, ArgType.TF_DTYPE)
            try:
                value = eval(signature.class_name)
            except:
                value = None
            return TFArgument(value, ArgType.TF_OBJECT)
        if label == "raw":
            try:
                value = json.loads(signature["value"])
            except:
                value = signature["value"]
                pass
            if isinstance(value, bool):
                return TFArgument(value, ArgType.BOOL)
            if isinstance(value, int):
                return TFArgument(value, ArgType.INT)
            if isinstance(value, str):
                return TFArgument(value, ArgType.STR)
            if isinstance(value, float):
                return TFArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(
                        TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(
                        TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)

        if label == "tuple":
            try:
                value = json.loads(signature["value"])
                tuple_value = []
                for elem in value:
                    tuple_value.append(
                        TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label == "list":
            try:
                try:
                    value = json.loads(signature["value"])
                except:
                    value = signature["value"]
                list_value = []
                for elem in value:
                    list_value.append(
                        TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label in ["tensor", "KerasTensor", "variable", "nparray"]:
            if not ("shape" in signature.keys() and "dtype" in signature.keys()):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]
            dtype = TFArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            else:
                minv, maxv = 0, 1
                shape = [
                    1,
                ]
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)

        return TFArgument(None, ArgType.NULL)


class TFAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        # self.record = TFDatabase.get_specified_record(api_name)
        self.record = TFDatabase.get_rand_record(
            api_name) if record is None else record
        self.args = TFAPI.generate_args_from_record(self.record)
        if re.findall(r"(tensorflow\.)", api_name):
            _name = self.api.split(".")[-2:]
            _name = ".".join(_name)
            self.is_class = inspect.isclass(eval(_name))
        else:
            self.is_class = inspect.isclass(eval(self.api))

    """
    Mutate based on vulnerability rules
    """

    def each_arg_mutate(self, arg, partition):
        if do_type_mutation():
            arg.mutate_type()
        arg.each_arg_mutation(partition)

    def new_mutate_tf(self):
        for param_index, p in enumerate(self.args):
            arg = self.args[p]
            if do_type_mutation():
                arg.mutate_type()
            arg.new_mutation(self.args, param_index, p)

    def new_mutate_multiple(self, arg, r):
        if do_type_mutation():
            arg.mutate_type()
        arg.new_mutation_multiple(r)

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(
                    self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code_oracle(self, prefix="arg", oracle=OracleType.CRASH) -> str:

        if oracle == OracleType.CRASH:
            code = self.to_code(prefix=prefix, res_name=RESULT_KEY)
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.CUDA:
            cpu_code = self.to_code(
                prefix=prefix,
                res_name=RES_CPU_KEY,
                use_try=True,
                err_name=ERR_CPU_KEY,
                wrap_device=True,
                device_name="CPU",
            )
            gpu_code = self.to_diff_code(
                prefix=prefix,
                res_name=RES_GPU_KEY,
                use_try=True,
                err_name=ERR_GPU_KEY,
                wrap_device=True,
                device_name="GPU:0",
            )

            code = cpu_code + gpu_code
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.PRECISION:
            low_code = self.to_code(
                prefix=prefix,
                res_name=RES_LOW_KEY,
                low_precision=True,
                use_try=True,
                err_name=ERR_LOW_KEY,
                time_it=True,
                time_var=TIME_LOW_KEY,
            )
            high_code = self.to_diff_code(
                prefix=prefix,
                res_name=RES_HIGH_KEY,
                use_try=True,
                err_name=ERR_HIGH_KEY,
                time_it=True,
                time_var=TIME_HIGH_KEY,
            )
            code = low_code + high_code
            return self.wrap_try(code, ERROR_KEY)
        return ""

    @staticmethod
    def generate_args_from_record(record: dict):
        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures["Label"] == "list":
                    s = signatures["value"]
                    if isinstance(s, list):
                        signatures = s
            args = []
            if signatures == None:
                return args
            for signature in signatures:
                x = TFArgument.generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[key] = TFArgument(value, ArgType.LIST)
            elif key != "output_signature":
                args[key] = TFArgument.generate_arg_from_signature(record[key])
        return args

    def _to_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_code(f"{prefix}_{index}",
                                    low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str

    def _to_diff_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_diff_code(
                f"{prefix}_{index}", low_precision=low_precision
            )
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_diff_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str

    def to_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:

        inputs = None
        input_name = ""
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_arg_code(
            prefix=prefix, low_precision=low_precision
        )
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            arg_code += f"{cls_name} = {self.api}({arg_str})\n"
            if inputs:
                arg_code += inputs.to_code(input_name,
                                           low_precision=low_precision)
                res_code += f"out = {cls_name}(*{input_name})\n"
        else:
            res_code = f"out = {self.api}({arg_str})\n"

        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def to_diff_code(
        self, prefix="arg", res_name="", low_precision=False, **kwargs
    ) -> str:

        inputs = None
        input_name = ""
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_diff_arg_code(
            prefix=prefix, low_precision=low_precision
        )
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            res_code = f""
            if inputs:
                arg_code += inputs.to_diff_code(input_name,
                                                low_precision=low_precision)
                res_code += f"{cls_name}(*{input_name})\n"
        else:
            res_code = f"{self.api}({arg_str})\n"

        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def _to_res_code(self, res_name, arg_str, input_name=None, prefix="arg"):
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            if input_name:
                res_code += f"{cls_name}(*{input_name})\n"
        else:
            res_code = f"{self.api}({arg_str})\n"
        return res_code

    def _to_invocation_code(
        self,
        arg_code,
        res_code,
        use_try=False,
        err_name="",
        wrap_device=False,
        device_name="",
        time_it=False,
        time_var="",
        **kwargs,
    ) -> str:
        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    @staticmethod
    def wrap_try(code: str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f'except Exception as e:\n  print("Error:"+str(e))\n'
        return wrapped_code

    @staticmethod
    def wrap_device(code: str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code: str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f'{RES_KEY}["{time_var}"] = t_end - t_start\n'
        return wrapped_code


def test_tf_arg():
    arg = TFArgument(None, ArgType.TF_TENSOR, shape=[2, 2], dtype=tf.int64)
    arg.mutate_value()
    print(arg.to_code("var"))
    print(arg.to_code("var", True))


def test_tf_api():
    api_name = "tf.keras.layers.Conv2D"
    record = TFDatabase.get_rand_record(api_name)
    api = TFAPI(api_name, record)
    api.mutate()
    print(api.to_code_oracle(oracle=OracleType.CRASH))
    print(api.to_code_oracle(oracle=OracleType.CUDA))
    print(api.to_code_oracle(oracle=OracleType.PRECISION))


if __name__ == "__main__":
    # test_tf_arg()
    test_tf_api()
