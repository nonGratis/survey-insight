"""Cloud KMS token encryption adapter."""

from __future__ import annotations

import base64

from google.cloud import kms_v1


class KmsTokenCrypto:
    """Encrypt and decrypt OAuth tokens with a Cloud KMS symmetric key."""

    prefix = "gcp-kms:v1:"

    def __init__(
        self,
        key_name: str,
        client: kms_v1.KeyManagementServiceClient | None = None,
    ) -> None:
        if not key_name:
            raise ValueError("KMS key name is required.")
        self.key_name = key_name
        self.client = client or kms_v1.KeyManagementServiceClient()

    def encrypt(self, plaintext: str) -> str:
        response = self.client.encrypt(
            request={
                "name": self.key_name,
                "plaintext": plaintext.encode("utf-8"),
            }
        )
        encoded = base64.urlsafe_b64encode(response.ciphertext).decode("ascii")
        return self.prefix + encoded

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext.startswith(self.prefix):
            raise ValueError("Unsupported KMS ciphertext format.")
        encoded = ciphertext.removeprefix(self.prefix)
        raw_ciphertext = base64.urlsafe_b64decode(encoded.encode("ascii"))
        response = self.client.decrypt(
            request={
                "name": self.key_name,
                "ciphertext": raw_ciphertext,
            }
        )
        return response.plaintext.decode("utf-8")
