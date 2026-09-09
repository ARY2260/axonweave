class AxonWeaveError(RuntimeError):
    code = "AXW000"

class SubstrateNotInstalledError(AxonWeaveError):
    code = "AXW001"

class DatasetIntegrityError(AxonWeaveError):
    code = "AXW002"

class SchemaError(AxonWeaveError):
    code = "AXW003"

class UnsupportedDeviceError(AxonWeaveError):
    code = "AXW004"

class BiologicalAssumptionError(AxonWeaveError):
    code = "AXW005"

class BackendUnavailableError(AxonWeaveError):
    code = "AXW006"
