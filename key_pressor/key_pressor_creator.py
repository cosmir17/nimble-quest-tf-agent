from key_pressor.MacKeyPressor import MacKeyPressor
from key_pressor.WindowKeyPressor import WindowKeyPressor
from key_pressor.AbstractKeyPressor import AbstractKeyPressor
import platform


def create_key_pressor() -> AbstractKeyPressor:
    os_name = platform.system()
    if os_name == "Darwin":
        return MacKeyPressor()
    if os_name == "Linux" or os_name == "Windows":
        return WindowKeyPressor()
    else:
        raise NotImplementedError("OS can't be recognised")
