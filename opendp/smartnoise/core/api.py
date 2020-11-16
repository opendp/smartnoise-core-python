import ctypes

import sys
import os
import re
import platform

from . import api_pb2


class LibraryWrapper(object):

    def __init__(self):

        extension = {
            "linux": ".so",
            "win32": ".dll",
            "darwin": ".dylib"
        }.get(sys.platform)

        prefix = "" if sys.platform == "win32" else "lib"

        if not extension:
            raise Exception(f"smartnoise-core does not support {sys.platform}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(script_dir, "lib")
        lib_smartnoise_path = os.path.join(lib_dir, f"{prefix}smartnoise_ffi{extension}")

        self.lib_smartnoise = ctypes.cdll.LoadLibrary(lib_smartnoise_path)

        proto_argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]

        class ByteBuffer(ctypes.Structure):
            _fields_ = [
                ('len', ctypes.c_int64),
                ('data', ctypes.POINTER(ctypes.c_uint8))]

        # validator
        self.lib_smartnoise.accuracy_to_privacy_usage.argtypes = proto_argtypes
        self.lib_smartnoise.compute_privacy_usage.argtypes = proto_argtypes
        self.lib_smartnoise.expand_component.argtypes = proto_argtypes
        self.lib_smartnoise.get_properties.argtypes = proto_argtypes
        self.lib_smartnoise.generate_report.argtypes = proto_argtypes
        self.lib_smartnoise.privacy_usage_to_accuracy.argtypes = proto_argtypes
        self.lib_smartnoise.validate_analysis.argtypes = proto_argtypes

        self.lib_smartnoise.accuracy_to_privacy_usage.restype = ByteBuffer
        self.lib_smartnoise.compute_privacy_usage.restype = ByteBuffer
        self.lib_smartnoise.expand_component.restype = ByteBuffer
        self.lib_smartnoise.get_properties.restype = ByteBuffer
        self.lib_smartnoise.generate_report.restype = ByteBuffer
        self.lib_smartnoise.privacy_usage_to_accuracy.restype = ByteBuffer
        self.lib_smartnoise.validate_analysis.restype = ByteBuffer

        # runtime
        self.lib_smartnoise.release.restype = ByteBuffer
        self.lib_smartnoise.release.argtypes = proto_argtypes

        # ffi
        self.lib_smartnoise.smartnoise_destroy_bytebuffer.restype = ctypes.c_void_p
        self.lib_smartnoise.smartnoise_destroy_bytebuffer.argtypes = [ByteBuffer]

        # direct mechanism access
        # library must be compiled with these endpoints to use them
        try:
            self.lib_smartnoise.laplace_mechanism.restype = ctypes.c_double
            self.lib_smartnoise.laplace_mechanism.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool
            ]

            self.lib_smartnoise.gaussian_mechanism.restype = ctypes.c_double
            self.lib_smartnoise.gaussian_mechanism.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool
            ]

            self.lib_smartnoise.simple_geometric_mechanism.restype = ctypes.c_int64
            self.lib_smartnoise.simple_geometric_mechanism.argtypes = [
                ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool
            ]

            self.lib_smartnoise.snapping_mechanism.restype = ctypes.c_double
            self.lib_smartnoise.snapping_mechanism.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool
            ]

            self.lib_smartnoise.snapping_mechanism_binding.restype = ctypes.c_double
            self.lib_smartnoise.snapping_mechanism_binding.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool
            ]

        except AttributeError:
            pass

    def validate_analysis(self, analysis, release):
        """
        FFI Helper. Check if an analysis is differentially private, given a set of released values.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :return: A success or failure response
        """
        return _communicate(
            argument=api_pb2.RequestValidateAnalysis(analysis=analysis, release=release),
            function=self.lib_smartnoise.validate_analysis,
            response_type=api_pb2.ResponseValidateAnalysis,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def compute_privacy_usage(self, analysis, release):
        """
        FFI Helper. Compute the overall privacy usage of an analysis.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :return: A privacy usage response
        """
        return _communicate(
            argument=api_pb2.RequestComputePrivacyUsage(analysis=analysis, release=release),
            function=self.lib_smartnoise.compute_privacy_usage,
            response_type=api_pb2.ResponseComputePrivacyUsage,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def generate_report(self, analysis, release):
        """
        FFI Helper. Generate a json string with a summary/report of the Analysis and Release
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :return: A protobuf response containing a json summary string
        """

        return _communicate(
            argument=api_pb2.RequestGenerateReport(analysis=analysis, release=release),
            function=self.lib_smartnoise.generate_report,
            response_type=api_pb2.ResponseGenerateReport,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def accuracy_to_privacy_usage(self, privacy_definition, component, properties, accuracies, public_arguments):
        """
        FFI Helper. Estimate the privacy usage necessary to bound accuracy to a given value.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param privacy_definition: A descriptive object defining neighboring, distance definitions
        :param component: The component to compute accuracy for
        :param properties: Properties about all of the arguments to the component
        :param accuracies: A value and alpha to convert to privacy usage for each column
        :param public_arguments: Public inputs to the component (like lower/upper for snapping)
        :return: A privacy usage response
        """
        return _communicate(
            argument=api_pb2.RequestAccuracyToPrivacyUsage(
                privacy_definition=privacy_definition,
                component=component,
                properties=properties,
                accuracies=accuracies,
                public_arguments=public_arguments),
            function=self.lib_smartnoise.accuracy_to_privacy_usage,
            response_type=api_pb2.ResponseAccuracyToPrivacyUsage,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def privacy_usage_to_accuracy(self, privacy_definition, component, properties, public_arguments, alpha):
        """
        FFI Helper. Estimate the accuracy of the release of a component, based on a privacy usage.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param privacy_definition: A descriptive object defining neighboring, distance definitions
        :param component: The component to compute accuracy for
        :param properties: Properties about all of the arguments to the component
        :param public_arguments: Public inputs to the component (like lower/upper for snapping)
        :param alpha: Used to set the confidence level for the accuracy
        :return: Accuracy estimates
        """
        return _communicate(
            argument=api_pb2.RequestPrivacyUsageToAccuracy(
                privacy_definition=privacy_definition,
                component=component,
                properties=properties,
                public_arguments=public_arguments,
                alpha=alpha),
            function=self.lib_smartnoise.privacy_usage_to_accuracy,
            response_type=api_pb2.ResponsePrivacyUsageToAccuracy,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def get_properties(self, analysis, release, node_ids=None):
        """
        FFI Helper. Derive static properties for all components in the graph.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :param node_ids: An optional list of node ids to derive properties for
        :return: A dictionary of property sets, one set of properties per component
        """
        return _communicate(
            argument=api_pb2.RequestGetProperties(analysis=analysis, release=release, node_ids=node_ids),
            function=self.lib_smartnoise.get_properties,
            response_type=api_pb2.ResponseGetProperties,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def compute_release(self, analysis, release, stack_trace, filter_level):
        """
        FFI Helper. Evaluate an analysis and release the differentially private results.
        This function touches private data. It calls the runtime rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :param stack_trace: Set to False to suppress stack traces
        :param filter_level: Configures how much data should be included in the release
        :return: A response containing an updated release
        """
        return _communicate(
            argument=api_pb2.RequestRelease(
                analysis=analysis,
                release=release,
                stack_trace=stack_trace,
                filter_level=filter_level),
            function=self.lib_smartnoise.release,
            response_type=api_pb2.ResponseRelease,
            destroy=self.lib_smartnoise.smartnoise_destroy_bytebuffer)

    def laplace_mechanism(self, value, epsilon, sensitivity, enforce_constant_time):
        """
        Direct api to access the laplace mechanism.

        :param value: float scalar to privatize
        :param epsilon: float privacy parameter
        :param sensitivity: float L1 sensitivity
        :param enforce_constant_time: ensure all calls take the same elapsed time
        :return: privatized float
        """
        return self.lib_smartnoise.laplace_mechanism(
            ctypes.c_double(value),
            ctypes.c_double(epsilon),
            ctypes.c_double(sensitivity),
            ctypes.c_bool(enforce_constant_time))

    def gaussian_mechanism(self, value, epsilon, delta, sensitivity, enforce_constant_time):
        """
        Direct api to access the gaussian mechanism.

        :param value: float scalar to privatize
        :param epsilon: float privacy parameter
        :param delta: float privacy parameter
        :param sensitivity: float L2 sensitivity
        :param enforce_constant_time: ensure all calls take the same elapsed time
        :return: privatized float
        """
        return self.lib_smartnoise.gaussian_mechanism(
            ctypes.c_double(value),
            ctypes.c_double(epsilon),
            ctypes.c_double(delta),
            ctypes.c_double(sensitivity),
            ctypes.c_bool(False),
            ctypes.c_bool(enforce_constant_time))

    def analytic_gaussian_mechanism(self, value, epsilon, delta, sensitivity, enforce_constant_time):
        """
        Direct api to access the analytic gaussian mechanism.

        :param value: float scalar to privatize
        :param epsilon: float privacy parameter
        :param delta: float privacy parameter
        :param sensitivity: float L2 sensitivity
        :param enforce_constant_time: ensure all calls take the same elapsed time
        :return: privatized float
        """
        return self.lib_smartnoise.gaussian_mechanism(
            ctypes.c_double(value),
            ctypes.c_double(epsilon),
            ctypes.c_double(delta),
            ctypes.c_double(sensitivity),
            ctypes.c_bool(True),
            ctypes.c_bool(enforce_constant_time))

    def simple_geometric_mechanism(self, value, epsilon, sensitivity, min, max, enforce_constant_time):
        """
        Direct api to access the simple geometric mechanism.

        :param value: integer scalar to privatize
        :param epsilon: float privacy parameter
        :param sensitivity: float L1 sensitivity
        :param min: lower bound on the statistic
        :param max: upper bound on the statistic
        :param enforce_constant_time: ensure all calls take the same elapsed time
        :return: privatized integer
        """
        return self.lib_smartnoise.simple_geometric_mechanism(
            ctypes.c_int64(value),
            ctypes.c_double(epsilon),
            ctypes.c_double(sensitivity),
            ctypes.c_int64(min),
            ctypes.c_int64(max),
            ctypes.c_bool(enforce_constant_time))

    def snapping_mechanism(self, value, epsilon, sensitivity, min, max, enforce_constant_time, binding_probability=None):
        """
        Direct api to access the snapping mechanism.

        :param value: float scalar to privatize
        :param epsilon: float privacy parameter
        :param sensitivity: float L1 sensitivity
        :param min: lower bound on the statistic
        :param max: upper bound on the statistic
        :param enforce_constant_time: ensure all calls take the same elapsed time
        :param binding_probability: optional float to scale clamping bounds based on the probability of the clamp binding
        :return: privatized float
        """
        if binding_probability is None:
            return self.lib_smartnoise.snapping_mechanism(
                ctypes.c_double(value),
                ctypes.c_double(epsilon),
                ctypes.c_double(sensitivity),
                ctypes.c_double(min),
                ctypes.c_double(max),
                ctypes.c_bool(enforce_constant_time))
        else:
            return self.lib_smartnoise.snapping_mechanism_binding(
                ctypes.c_double(value),
                ctypes.c_double(epsilon),
                ctypes.c_double(sensitivity),
                ctypes.c_double(min),
                ctypes.c_double(max),
                ctypes.c_double(binding_probability),
                ctypes.c_bool(enforce_constant_time))


def _communicate(function, destroy, argument, response_type):
    """
    Call the function with the proto argument, over ffi. Deserialize the response and optionally throw an error.

    :param function: function from lib_*
    :param destroy: function to destroy the bytebuffer returned from the function
    :param argument: proto object from api.proto
    :param response_type: proto object from api.proto
    :return: the .data field of the protobuf response
    """
    serialized_argument = argument.SerializeToString()

    bytes_array = bytearray(serialized_argument)
    buffer = (ctypes.c_ubyte * len(serialized_argument)).from_buffer(bytes_array)

    byte_buffer = function(buffer, len(serialized_argument))
    serialized_response = ctypes.string_at(byte_buffer.data, byte_buffer.len)

    response = response_type.FromString(serialized_response)

    destroy(byte_buffer)

    # Errors from here are propagated up from either the rust validator or runtime
    if response.HasField("error"):
        library_traceback = format_error(response.error)

        # stack traces beyond this point come from the internal rust libraries
        raise RuntimeError(library_traceback)
    return response.data


def format_error(error):
    library_traceback = error.message

    # noinspection PyBroadException
    try:
        # on Linux, stack traces are more descriptive
        if platform.system() == "Linux":
            message, *frames = re.split("\n +[0-9]+: ", library_traceback)
            library_traceback = '\n'.join(reversed(["  " + frame.replace("         at", "at") for frame in frames
                                                    if ("at src/" in frame or "smartnoise_validator" in frame)
                                                    and "smartnoise_validator::errors::Error" not in frame])) \
                                + "\n  " + message
    except Exception:
        pass

    return library_traceback
