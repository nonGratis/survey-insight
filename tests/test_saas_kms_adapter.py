from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.saas.adapters.kms import KmsTokenCrypto


@dataclass(frozen=True)
class _EncryptResponse:
    ciphertext: bytes


@dataclass(frozen=True)
class _DecryptResponse:
    plaintext: bytes


class _FakeKmsClient:
    def __init__(self) -> None:
        self.encrypt_requests: list[dict] = []
        self.decrypt_requests: list[dict] = []

    def encrypt(self, request: dict) -> _EncryptResponse:
        self.encrypt_requests.append(request)
        plaintext = request["plaintext"]
        return _EncryptResponse(ciphertext=b"encrypted:" + plaintext[::-1])

    def decrypt(self, request: dict) -> _DecryptResponse:
        self.decrypt_requests.append(request)
        ciphertext = request["ciphertext"]
        if not ciphertext.startswith(b"encrypted:"):
            raise ValueError("bad ciphertext")
        return _DecryptResponse(plaintext=ciphertext.removeprefix(b"encrypted:")[::-1])


def test_kms_token_crypto_round_trips_without_exposing_plaintext() -> None:
    client = _FakeKmsClient()
    crypto = KmsTokenCrypto("projects/p/locations/global/keyRings/r/cryptoKeys/k", client=client)

    ciphertext = crypto.encrypt("refresh-token")

    assert ciphertext.startswith("gcp-kms:v1:")
    assert "refresh-token" not in ciphertext
    assert client.encrypt_requests[0]["name"].endswith("/cryptoKeys/k")
    assert crypto.decrypt(ciphertext) == "refresh-token"


def test_kms_token_crypto_rejects_unknown_ciphertext_prefix() -> None:
    crypto = KmsTokenCrypto(
        "projects/p/locations/global/keyRings/r/cryptoKeys/k", client=_FakeKmsClient()
    )

    with pytest.raises(ValueError, match="Unsupported KMS ciphertext"):
        crypto.decrypt("fake-kms:abc")


def test_kms_token_crypto_requires_key_name() -> None:
    with pytest.raises(ValueError, match="KMS key name"):
        KmsTokenCrypto("", client=_FakeKmsClient())
