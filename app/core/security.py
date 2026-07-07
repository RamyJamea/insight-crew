import base64
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

current_file = Path(__file__)
parent_dir = current_file.parent
grandparent_dir = current_file.parent.parent
key_path = grandparent_dir / "private_key.pem"
with open(key_path, "rb") as f:
    key_data = f.read()


with open(key_path, "rb") as f:
    private_key = serialization.load_pem_private_key(
        f.read(), password=None, backend=default_backend()
    )


def get_llm_config(model_payload: str):
    cipher_bytes = base64.b64decode(model_payload)

    decrypted_bytes = private_key.decrypt(
        cipher_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    provider, model, key = decrypted_bytes.decode("utf-8").split("#")
    return provider, model, key
