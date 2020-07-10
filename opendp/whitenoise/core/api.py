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

        if not extension:
            raise Exception(f"whitenoise-core does not support {sys.platform}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(script_dir, "lib")
        lib_validator_path = os.path.join(lib_dir, "libwhitenoise_validator" + extension)
        lib_runtime_path = os.path.join(lib_dir, "libwhitenoise_runtime" + extension)

        self.lib_validator = ctypes.cdll.LoadLibrary(lib_validator_path)
        self.lib_runtime = ctypes.cdll.LoadLibrary(lib_runtime_path)

        class ByteBuffer(ctypes.Structure):
            _fields_ = [
                ('len', ctypes.c_int64),
                ('data', ctypes.POINTER(ctypes.c_uint8))]

        # validator
        self.lib_validator.accuracy_to_privacy_usage.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.compute_privacy_usage.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.expand_component.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.get_properties.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.generate_report.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.privacy_usage_to_accuracy.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.validate_analysis.argtypes = LibraryWrapper._get_argtypes()
        self.lib_validator.whitenoise_validator_destroy_bytebuffer.argtypes = [ByteBuffer]

        self.lib_validator.accuracy_to_privacy_usage.restype = ByteBuffer
        self.lib_validator.compute_privacy_usage.restype = ByteBuffer
        self.lib_validator.expand_component.restype = ByteBuffer
        self.lib_validator.get_properties.restype = ByteBuffer
        self.lib_validator.generate_report.restype = ByteBuffer
        self.lib_validator.privacy_usage_to_accuracy.restype = ByteBuffer
        self.lib_validator.validate_analysis.restype = ByteBuffer
        self.lib_validator.whitenoise_validator_destroy_bytebuffer.restype = ctypes.c_void_p

        # runtime
        self.lib_runtime.release.restype = ByteBuffer
        self.lib_runtime.whitenoise_runtime_destroy_bytebuffer.restype = ctypes.c_void_p

        self.lib_runtime.release.argtypes = LibraryWrapper._get_argtypes()
        self.lib_runtime.whitenoise_runtime_destroy_bytebuffer.argtypes = [ByteBuffer]

    @staticmethod
    def _get_argtypes():
        return [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]

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
            function=self.lib_validator.validate_analysis,
            response_type=api_pb2.ResponseValidateAnalysis,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

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
            function=self.lib_validator.compute_privacy_usage,
            response_type=api_pb2.ResponseComputePrivacyUsage,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

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
            function=self.lib_validator.generate_report,
            response_type=api_pb2.ResponseGenerateReport,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

    def accuracy_to_privacy_usage(self, privacy_definition, component, properties, accuracies):
        """
        FFI Helper. Estimate the privacy usage necessary to bound accuracy to a given value.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param privacy_definition: A descriptive object defining neighboring, distance definitions
        :param component: The component to compute accuracy for
        :param properties: Properties about all of the arguments to the component
        :param accuracies: A value and alpha to convert to privacy usage for each column
        :return: A privacy usage response
        """
        return _communicate(
            argument=api_pb2.RequestAccuracyToPrivacyUsage(
                privacy_definition=privacy_definition,
                component=component,
                properties=properties,
                accuracies=accuracies),
            function=self.lib_validator.accuracy_to_privacy_usage,
            response_type=api_pb2.ResponseAccuracyToPrivacyUsage,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

    def privacy_usage_to_accuracy(self, privacy_definition, component, properties, alpha):
        """
        FFI Helper. Estimate the accuracy of the release of a component, based on a privacy usage.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param privacy_definition: A descriptive object defining neighboring, distance definitions
        :param component: The component to compute accuracy for
        :param properties: Properties about all of the arguments to the component
        :param alpha: Used to set the confidence level for the accuracy
        :return: Accuracy estimates
        """
        return _communicate(
            argument=api_pb2.RequestPrivacyUsageToAccuracy(
                privacy_definition=privacy_definition,
                component=component,
                properties=properties,
                alpha=alpha),
            function=self.lib_validator.privacy_usage_to_accuracy,
            response_type=api_pb2.ResponsePrivacyUsageToAccuracy,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

    def get_properties(self, analysis, release):
        """
        FFI Helper. Derive static properties for all components in the graph.
        This function is data agnostic. It calls the validator rust FFI with protobuf objects.

        :param analysis: A description of computation
        :param release: A collection of public values
        :return: A dictionary of property sets, one set of properties per component
        """
        return _communicate(
            argument=api_pb2.RequestGetProperties(analysis=analysis, release=release),
            function=self.lib_validator.get_properties,
            response_type=api_pb2.ResponseGetProperties,
            destroy=self.lib_validator.whitenoise_validator_destroy_bytebuffer)

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
            function=self.lib_runtime.release,
            response_type=api_pb2.ResponseRelease,
            destroy=self.lib_runtime.whitenoise_runtime_destroy_bytebuffer)


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
                                                    if ("at src/" in frame or "whitenoise_validator" in frame)
                                                    and "whitenoise_validator::errors::Error" not in frame])) \
                                + "\n  " + message
    except Exception:
        pass

    return library_traceback
